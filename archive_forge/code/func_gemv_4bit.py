import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def gemv_4bit(A: Tensor, B: Tensor, out: Optional[torch.Tensor]=None, transposed_A=False, transposed_B=False, state=None):
    prev_device = pre_call(A.device)
    if state is None:
        raise ValueError('state cannot None. gem_4bit( ) requires the state from quantize_4bit( )')
    if A.numel() != A.shape[-1]:
        raise ValueError('Dimensions of A are invalid. Must be a vector with the leading dimensions of "1", e.g. [1, 1, 2048]')
    Bshape = state.shape
    bout = Bshape[0]
    absmax = state.absmax
    if state.nested:
        absmax = dequantize_blockwise(state.absmax, state.state2)
        absmax += state.offset
    if out is None:
        if len(A.shape) == 3:
            out = torch.empty(size=(A.shape[0], A.shape[1], bout), dtype=A.dtype, device=A.device)
        else:
            out = torch.empty(size=(A.shape[0], bout), dtype=A.dtype, device=A.device)
    n = 1
    m = Bshape[0]
    k = Bshape[1]
    lda = Bshape[0]
    ldc = Bshape[0]
    ldb = (A.shape[-1] + 1) // 2
    is_on_gpu([B, A, out, absmax, state.code])
    m = ct.c_int32(m)
    n = ct.c_int32(n)
    k = ct.c_int32(k)
    lda = ct.c_int32(lda)
    ldb = ct.c_int32(ldb)
    ldc = ct.c_int32(ldc)
    if B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32]:
        if A.dtype == torch.float16:
            lib.cgemm_4bit_inference_naive_fp16(m, n, k, get_ptr(A), get_ptr(B), get_ptr(absmax), get_ptr(state.code), get_ptr(out), lda, ldb, ldc, ct.c_int32(state.blocksize))
        elif A.dtype == torch.bfloat16:
            lib.cgemm_4bit_inference_naive_bf16(m, n, k, get_ptr(A), get_ptr(B), get_ptr(absmax), get_ptr(state.code), get_ptr(out), lda, ldb, ldc, ct.c_int32(state.blocksize))
        elif A.dtype == torch.float32:
            lib.cgemm_4bit_inference_naive_fp32(m, n, k, get_ptr(A), get_ptr(B), get_ptr(absmax), get_ptr(state.code), get_ptr(out), lda, ldb, ldc, ct.c_int32(state.blocksize))
        else:
            raise NotImplementedError(f'Matmul not implemented for data type {A.dtype}')
    else:
        raise NotImplementedError(f'Matmul not implemented for data type {A.dtype}')
    post_call(prev_device)
    return out