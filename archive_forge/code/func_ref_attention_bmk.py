import itertools
import random
from functools import partial
import torch
from torch.utils import benchmark
import xformers.ops
import xformers.ops.fmha as fmha
from xformers.attn_bias_utils import create_attn_bias
from xformers.benchmarks.utils import benchmark_main_helper
def ref_attention_bmk(q, k, v, attn_bias=None, p=0.0):
    if isinstance(attn_bias, xformers.ops.AttentionMask):
        attn_bias = attn_bias.materialize((q.shape[0], 1, q.shape[1], k.shape[1])).to(q).squeeze()
    q = q * (1.0 / q.shape[-1] ** 0.5)
    if attn_bias is None:
        attn = q @ k.transpose(-2, -1)
    else:
        attn = torch.baddbmm(attn_bias, q, k.transpose(-2, -1))
    attn = attn.softmax(-1)
    if p > 0:
        attn = torch.nn.functional.dropout(attn, p=p)
    return attn @ v