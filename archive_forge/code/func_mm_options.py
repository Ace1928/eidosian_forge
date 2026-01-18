import functools
import logging
from typing import cast, List, Tuple
import sympy
import torch
from torch._inductor.select_algorithm import realize_inputs
from torch._inductor.virtualized import V
from ..utils import ceildiv as cdiv, next_power_of_2
def mm_options(config, sym_k, layout, b_prologue_cast_type=None):
    """
    Common options to matmul triton templates.
    """
    even_k_symbolic = sympy.gcd(sym_k, config.kwargs['BLOCK_K']) == config.kwargs['BLOCK_K']
    return dict(GROUP_M=8, EVEN_K=even_k_symbolic, ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32, ACC_TYPE=acc_type(layout.dtype), B_PROLOGUE_CAST_TYPE=b_prologue_cast_type, num_stages=config.num_stages, num_warps=config.num_warps, **config.kwargs)