
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 引入所有可能的多项式库
import numpy.polynomial.chebyshev as cheb
import numpy.polynomial.legendre as leg
import numpy.polynomial.hermite as herm
import numpy.polynomial.polynomial as poly

class DecomposedSynthesizer():
    """
    Modeling: Data = (Temporal_Basis * Coeffs) * Sigma * Spatial_Basis^T
    """

    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel  # 1
        self.time_steps = hs  # 2000
        self.subcarriers = ws  # 30
        self.device = device
        self.model = None

        # --- 超参数设置 ---
        self.rank = args.rank
        self.seg_len = args.seglen
        self.degree = args.degree

        # [新增] 获取基底类型，默认为 chebyshev
        self.basis_type = getattr(args, 'basis', 'chebyshev').lower()

        # 校验
        if self.time_steps % self.seg_len != 0:
            raise ValueError(f"Time steps {self.time_steps} must be divisible by seg_len {self.seg_len}")
        self.num_segments = self.time_steps // self.seg_len
        self.num_coeffs = self.degree + 1

        print(f"\n[Decomposed Synthesizer] Rank: {self.rank}, Segs: {self.num_segments}, Degree: {self.degree}")
        print(f"[Basis Type] Using: {self.basis_type}")

        # --- 1. 预计算时间维基底矩阵 (固定) ---
        # Basis shape: (num_coeffs, seg_len) -> e.g. (11, 200)
        basis_numpy = self._get_basis_matrix(self.basis_type, self.seg_len, self.degree)

        self.temporal_basis = torch.from_numpy(basis_numpy).float().to(self.device)

        # 预计算伪逆矩阵，用于初始化时的投影: Coeffs = U * Basis_pinv
        # shape: (seg_len, num_coeffs) -> (200, 11)
        basis_pinv = np.linalg.pinv(basis_numpy)
        self.temporal_basis_pinv = torch.from_numpy(basis_pinv).float().to(self.device)

        # --- 2. 定义可学习参数 ---
        num_syn_data = self.nclass * self.ipc

        # A. 时间维系数 (通用命名)
        # 初始化为一个很小的值，具体值会在 init() 中被覆盖
        self.temporal_coeffs = nn.Parameter(torch.randn(
            num_syn_data, self.rank, self.num_segments, self.num_coeffs, device=self.device
        ) * 0.01)

        # B. 奇异值
        self.sigma = nn.Parameter(torch.rand(num_syn_data, self.rank, device=self.device))

        # C. 空间/信道基
        self.v_channel = nn.Parameter(torch.randn(
            num_syn_data, self.subcarriers, self.rank, device=self.device
        ) * 0.01)

        # 标签
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.int64,
                                    requires_grad=False,
                                    device=self.device).view(-1)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type

        # 模型加载逻辑 (保持不变)
        from common import define_model
        self.model = define_model(args, nclass).to(device)
        if args.num_premodel > 0:
            import random, os
            slkt_model_id = random.randint(0, args.num_premodel - 1)
            final_path = os.path.join(args.pretrain_dir, 'premodel{}_trained.pth.tar'.format(slkt_model_id))
            # 兼容性处理
            if os.path.exists(final_path):
                self.model.load_state_dict(torch.load(final_path))
                print(f"Loaded pre-trained model: {final_path}")
            self.model.eval()

        print(f"Total Params: {self.temporal_coeffs.numel() + self.sigma.numel() + self.v_channel.numel()}")

    def _get_basis_matrix(self, basis_type, length, degree):
        """根据类型生成基底矩阵，返回 shape: (degree+1, length)"""
        # 定义域 [-1, 1] 通常是大多数正交多项式的最佳区间
        x = np.linspace(-1, 1, length)

        if basis_type == 'chebyshev':
            # 切比雪夫 (最佳边缘拟合)
            return cheb.chebvander(x, degree).T

        elif basis_type == 'legendre':
            # 勒让德 (最平滑，类似球谐基)
            return leg.legvander(x, degree).T

        elif basis_type == 'hermite':
            # 埃尔米特 (物理学基，带高斯权重倾向)
            return herm.hermvander(x, degree).T

        elif basis_type == 'polynomial':
            # 标准幂基 (1, x, x^2...) - 用于展示"难优化"的情况
            return poly.polyvander(x, degree).T

        else:
            raise ValueError(f"Unsupported basis type: {basis_type}")

    def reconstruct_data(self):
        """ X = U * S * V^T """
        # 1. U (Time): Coeffs @ Basis
        # (N, Rank, Segs, Coeffs) @ (Coeffs, SegLen) -> (N, Rank, Segs, SegLen)
        u_segments = torch.matmul(self.temporal_coeffs, self.temporal_basis)

        # Flatten segments: (N, Rank, Time)
        u_time = u_segments.view(u_segments.shape[0], self.rank, -1)

        # 2. S (Sigma)
        s_diag = self.sigma.unsqueeze(-1)  # (N, Rank, 1)

        # 3. V (Space)
        v_t = self.v_channel.permute(0, 2, 1)  # (N, Rank, 30)

        # 4. Combine: (U * S) @ V^T
        weighted_u = (u_time * s_diag).permute(0, 2, 1)  # (N, 2000, Rank)
        reconstructed = torch.matmul(weighted_u, v_t)  # (N, 2000, 30)

        return reconstructed.unsqueeze(1)  # (N, 1, 2000, 30)

    @property
    def data(self):
        return self.reconstruct_data()

    def parameters(self):
        return [self.temporal_coeffs, self.sigma, self.v_channel]

    def init(self, dataset, loader, init_type='dream'):

        print(f"Initializing with strategy: {init_type} (Basis: {self.basis_type})...")

        all_selected_images = []

        # 遍历类别获取真实数据
        for c in range(self.nclass):
            if hasattr(loader, 'class_sample'):
                n_total_c = len(loader.class_indices[c]) if hasattr(loader, 'class_indices') else 2000
                img_real, _ = loader.class_sample(c, n_total_c)
            else:
                # 简单 Fallback，如果不使用 class_sample
                indices = [i for i, label in enumerate(dataset.targets) if label == c]
                # 随机采样一些
                sel_indices = np.random.choice(indices, min(len(indices), 500), replace=False)
                img_real = torch.stack([dataset[i][0] for i in sel_indices]).to(self.device)
            self.model.eval()
            strategy = NEW_Strategy(img_real, self.model)
            query_idxs = strategy.query_no_pca(self.ipc)  # 获取 IPC 个最佳样本
            sel_img = img_real[query_idxs].detach()
            all_selected_images.append(sel_img)

            if c == 0:
                print(f"Class 0 selected {len(query_idxs)} samples.")

        # 拼接
        selected_tensor = torch.cat(all_selected_images, dim=0).to(self.device)
        if selected_tensor.dim() == 4:
            selected_tensor = selected_tensor.squeeze(1)  # (N, 2000, 30)

        # 执行分解 (SVD + Temporal Basis Projection)
        with torch.no_grad():
            # A. SVD 分解
            U, S, Vh = torch.linalg.svd(selected_tensor, full_matrices=False)

            # B. 截断保留前 Rank 个分量
            U_k = U[:, :, :self.rank]  # (N, 2000, R)
            S_k = S[:, :self.rank]  # (N, R)
            Vh_k = Vh[:, :self.rank, :]  # (N, R, 30)

            # C. 赋值 Sigma
            self.sigma.data.copy_(S_k)

            # D. 赋值 V_channel (Spatial Basis)
            # Vh 是 V^T，我们要存储 V (N, 30, R)，即 Vh 的转置
            self.v_channel.data.copy_(Vh_k.permute(0, 2, 1))

            # E. 计算时间维多项式系数 (Temporal Basis Coeffs)
            # U_k shape: (N, 2000, R)
            # 1. 重排为分段格式: (N, R, Num_Segs, Seg_Len)
            u_reshaped = U_k.permute(0, 2, 1).contiguous().view(
                self.nclass * self.ipc, self.rank, self.num_segments, self.seg_len
            )

            # 2. 投影求解系数: Coeffs = Data @ Basis_PseudoInverse
            # Data: (..., SegLen)
            # Basis_pinv: (SegLen, NumCoeffs)
            # Result: (..., NumCoeffs)
            coeffs_init = torch.matmul(u_reshaped, self.temporal_basis_pinv)

            # F. 赋值 Coeffs
            self.temporal_coeffs.data.copy_(coeffs_init)

        print(f"Initialization complete using {self.basis_type} basis.")

    def sample(self, c, max_size=128):
        """Sample synthetic data per class"""
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)

        coeffs = self.temporal_coeffs[idx_from:idx_to]
        sigma = self.sigma[idx_from:idx_to]
        v = self.v_channel[idx_from:idx_to]

        # 局部重构
        u_segments = torch.matmul(coeffs, self.temporal_basis)
        u_time = u_segments.view(coeffs.shape[0], self.rank, -1)

        s_diag = sigma.unsqueeze(-1)
        weighted_u = (u_time * s_diag).permute(0, 2, 1)
        v_t = v.permute(0, 2, 1)

        reconstructed = torch.matmul(weighted_u, v_t).unsqueeze(1)

        return reconstructed, self.targets[idx_from:idx_to]

    # loader 和 test 函数保持不变 ...
    def loader(self, args, augment=True):
        full_data = self.reconstruct_data().detach().cpu()
        full_targets = self.targets.detach().cpu()
        train_dataset = TensorDataset(full_data, full_targets)
        nw = 0 if not augment else args.workers
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)
        return train_loader

    def test(self, args, val_loader, logger, bench=True):
        loader = self.loader(args, args.augment)
        from test import test_data
        convnet_result = test_data(args, loader, val_loader, test_resnet=False, logger=logger)
        return convnet_result