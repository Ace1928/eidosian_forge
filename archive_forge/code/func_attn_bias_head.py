import itertools
import random
from functools import partial
import torch
from torch.utils import benchmark
import xformers.ops
import xformers.ops.fmha as fmha
from xformers.attn_bias_utils import create_attn_bias
from xformers.benchmarks.utils import benchmark_main_helper
def attn_bias_head(head: int):
    if isinstance(attn_bias, torch.Tensor):
        assert attn_bias.ndim == 4
        _, H, _, _ = attn_bias.shape
        assert H == Hq
        bias_bghmn = attn_bias.reshape(B, Hkv, nhead_ratio_qk, M, N)
        return bias_bghmn[:, :, head]
    if isinstance(attn_bias, fmha.attn_bias.LowerTriangularMaskWithTensorBias):
        assert attn_bias._bias.ndim == 4
        _, H, _, _ = attn_bias._bias.shape
        assert H == Hq
        bias_bghmn = attn_bias._bias.reshape(B, Hkv, nhead_ratio_qk, M, N)
        return fmha.attn_bias.LowerTriangularMaskWithTensorBias(bias_bghmn[:, :, head])
    return attn_bias