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
def igemm(A: Tensor, B: Tensor, out: Optional[torch.Tensor]=None, transposed_A=False, transposed_B=False):
    sout = check_matmul(A, B, out, transposed_A, transposed_B)
    if out is None:
        out = torch.zeros(size=sout, dtype=torch.int32, device=A.device)
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1]:
            return batched_igemm(A, B, out)
    sA = A.shape
    sB = B.shape
    if transposed_A and len(sA) == 2:
        sA = (sA[1], sA[0])
    elif transposed_A and len(sA) == 3:
        sA = (sA[0], sA[2], sA[0])
    if transposed_B and len(sB) == 2:
        sB = (sB[1], sB[0])
    elif transposed_B and len(sB) == 3:
        sB = (sB[0], sB[2], sB[0])
    if len(sB) == 2:
        if B.stride()[0] == B.shape[1]:
            transposed_B = False
        elif B.stride()[1] == B.shape[0]:
            transposed_B = True
        if len(A.shape) == 2:
            if A.stride()[0] == A.shape[1]:
                transposed_A = False
            elif A.stride()[1] == A.shape[0]:
                transposed_A = True
        elif A.stride()[1] == A.shape[2]:
            transposed_A = False
        elif A.stride()[2] == A.shape[1]:
            transposed_A = True
        if len(sA) == 2:
            n = sA[0]
            ldb = A.stride()[1 if transposed_A else 0]
        elif len(sA) == 3 and len(sB) == 2:
            n = sA[0] * sA[1]
            ldb = sA[2]
        m = sB[1]
        k = sB[0]
        lda = B.stride()[1 if transposed_B else 0]
        ldc = sB[1]
    elif len(sB) == 3:
        assert len(sA) == 3
        if not (sA[0] == sB[0] and sA[1] == sB[1]):
            raise ValueError(f'Only bsi,bso->io supported for tensor contractions, but dims for A x B were: {sA} x {sB}')
        transposed_A = True
        transposed_B = False
        m = sB[2]
        n = sA[2]
        k = sB[0] * sB[1]
        lda = m
        ldb = sA[2]
        ldc = m
    ptr = CUBLAS_Context.get_instance().get_context(A.device)
    is_on_gpu([B, A, out])
    lib.cigemm(ptr, ct.c_bool(transposed_B), ct.c_bool(transposed_A), ct.c_int32(m), ct.c_int32(n), ct.c_int32(k), get_ptr(B), get_ptr(A), get_ptr(out), ct.c_int32(lda), ct.c_int32(ldb), ct.c_int32(ldc))
    return out