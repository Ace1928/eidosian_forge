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
def spmm_coo_very_sparse(cooA, B, dequant_stats=None, out=None):
    if out is None:
        out = torch.zeros((cooA.rows, B.shape[1]), device=B.device, dtype=cooA.values.dtype)
    nnz = cooA.nnz
    prev_device = pre_call(B.device)
    assert cooA.rowidx.numel() == nnz
    assert cooA.colidx.numel() == nnz
    assert cooA.values.numel() == nnz
    assert cooA.cols == B.shape[0], f'{cooA.cols} vs {B.shape}'
    transposed_B = False if B.is_contiguous() else True
    ldb = B.stride()[1 if transposed_B else 0]
    ldc = B.shape[1]
    values, counts = torch.unique(cooA.rowidx, return_counts=True)
    offset = counts.cumsum(0).int()
    max_count, max_idx = torch.sort(counts, descending=True)
    max_idx = max_idx.int()
    max_count = max_count.int()
    assert max_count[0] <= 32, f'Current max count per row is 8 but found {max_count[0]}.'
    assert B.dtype in [torch.float16, torch.int8]
    ptrOffset = get_ptr(offset)
    ptrMaxCount = get_ptr(max_count)
    ptrMaxIdx = get_ptr(max_idx)
    ptrRowidx = get_ptr(cooA.rowidx)
    ptrColidx = get_ptr(cooA.colidx)
    ptrValues = get_ptr(cooA.values)
    ptrB = get_ptr(B)
    ptrC = get_ptr(out)
    ptrDequantStats = get_ptr(dequant_stats)
    cnnz_rows = ct.c_int32(counts.numel())
    cnnz = ct.c_int32(cooA.nnz)
    crowsA = ct.c_int32(cooA.rows)
    ccolsA = ct.c_int32(cooA.cols)
    crowsB = ct.c_int32(B.shape[1])
    ccolsB = ct.c_int32(B.shape[1])
    cldb = ct.c_int32(ldb)
    cldc = ct.c_int32(ldc)
    is_on_gpu([cooA.rowidx, cooA.colidx, cooA.values, B, out, dequant_stats])
    if B.dtype == torch.float16:
        lib.cspmm_coo_very_sparse_naive_fp16(ptrMaxCount, ptrMaxIdx, ptrOffset, ptrRowidx, ptrColidx, ptrValues, ptrB, ptrC, ptrDequantStats, cnnz_rows, cnnz, crowsA, crowsB, ccolsB)
    elif B.dtype == torch.int8:
        lib.cspmm_coo_very_sparse_naive_int8(ptrMaxCount, ptrMaxIdx, ptrOffset, ptrRowidx, ptrColidx, ptrValues, ptrB, ptrC, ptrDequantStats, cnnz_rows, cnnz, crowsA, crowsB, ccolsB)
    post_call(prev_device)
    return out