import torch
from ..lowering import register_lowering
from ..select_algorithm import (
from ..utils import ceildiv as cdiv, use_aten_gemm_kernels, use_triton_template
from .mm_common import addmm_epilogue, mm_args, mm_configs, mm_options
def bmm_grid(b, m, n, meta):
    return (cdiv(m, meta['BLOCK_M']) * cdiv(n, meta['BLOCK_N']), b, 1)