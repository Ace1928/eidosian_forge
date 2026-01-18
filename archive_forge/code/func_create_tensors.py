import itertools
import random
from functools import partial
import torch
from torch.utils import benchmark
import xformers.ops
import xformers.ops.fmha as fmha
from xformers.attn_bias_utils import create_attn_bias
from xformers.benchmarks.utils import benchmark_main_helper
def create_tensors(shape, dtype, requires_grad=False, packed=True, multiquery=False):
    stacked_shape = list(shape)
    stacked_dim = 2 if packed else 0
    stacked_shape.insert(stacked_dim, 3)
    qkv = torch.rand(stacked_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    q = torch.rand(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    shape_kv = (shape[0], shape[1], 1 if multiquery else shape[2], shape[3])
    k = torch.rand(shape_kv, device=device, dtype=dtype, requires_grad=requires_grad).expand(shape)
    v = torch.rand(shape_kv, device=device, dtype=dtype, requires_grad=requires_grad).expand(shape)
    return (qkv, q, k, v)