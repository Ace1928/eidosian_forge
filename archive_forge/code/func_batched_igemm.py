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
def batched_igemm(A: Tensor, B: Tensor, out: Optional[torch.Tensor]=None, transposed_A=False, transposed_B=False):
    if not len(A.shape) == 3 or not len(B.shape) == 3:
        raise ValueError(f'Expected 3-dimensional tensors for bmm, but got shapes A and B: {A.shape} and {B.shape}')
    sout = check_matmul(A, B, out, transposed_A, transposed_B)
    if out is None:
        out = torch.zeros(size=sout, dtype=torch.int32, device=A.device)
    if B.is_contiguous():
        lda = B.stride()[1]
        transposed_A = False
    else:
        s = B.stride()
        if s[0] != B.shape[0]:
            B = B.contiguous()
            lda = B.stride()[1]
        elif s[2] == B.shape[1]:
            transposed_A = True
            lda = B.stride()[2]
        elif s[2] == 1:
            B = B.contiguous()
            lda = B.stride()[1]
        elif s[1] == 1:
            B = B.contiguous()
            lda = B.stride()[1]
        else:
            B = B.contiguous()
            lda = B.stride()[1]
    if A.is_contiguous():
        ldb = A.stride()[1]
        transposed_B = False
    else:
        s = A.stride()
        if s[0] != A.shape[0]:
            A = A.contiguous()
            ldb = A.stride()[1]
            transposed_B = False
        elif s[2] == A.shape[1]:
            ldb = A.stride()[2]
            transposed_B = True
        else:
            A = A.contiguous()
            ldb = A.stride()[1]
            transposed_B = False
    num_batch = A.shape[0]
    n = A.shape[1]
    m = B.shape[2]
    k = B.shape[1]
    ldc = m
    strideA = B.shape[1] * B.shape[2]
    strideB = A.shape[1] * A.shape[2]
    strideC = A.shape[1] * B.shape[2]
    ptr = CUBLAS_Context.get_instance().get_context(A.device)
    is_on_gpu([B, A, out])
    lib.cbatched_igemm(ptr, ct.c_bool(transposed_B), ct.c_bool(transposed_A), ct.c_int32(m), ct.c_int32(n), ct.c_int32(k), get_ptr(B), get_ptr(A), get_ptr(out), ct.c_int32(lda), ct.c_int32(ldb), ct.c_int32(ldc), ct.c_long(strideA), ct.c_long(strideB), ct.c_long(strideC), ct.c_uint32(num_batch))
    return out