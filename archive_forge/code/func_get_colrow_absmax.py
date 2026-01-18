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
def get_colrow_absmax(A, row_stats=None, col_stats=None, nnz_block_ptr=None, threshold=0.0):
    assert A.dtype == torch.float16
    device = A.device
    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]
    col_tiles = (cols + 255) // 256
    tiled_rows = (rows + 15) // 16 * 16
    if row_stats is None:
        row_stats = torch.empty((rows,), dtype=torch.float32, device=device).fill_(-50000.0)
    if col_stats is None:
        col_stats = torch.empty((cols,), dtype=torch.float32, device=device).fill_(-50000.0)
    if nnz_block_ptr is None and threshold > 0.0:
        nnz_block_ptr = torch.zeros((tiled_rows * col_tiles + 1,), dtype=torch.int32, device=device)
    ptrA = get_ptr(A)
    ptrRowStats = get_ptr(row_stats)
    ptrColStats = get_ptr(col_stats)
    ptrNnzrows = get_ptr(nnz_block_ptr)
    rows = ct.c_int32(rows)
    cols = ct.c_int32(cols)
    prev_device = pre_call(A.device)
    is_on_gpu([A, row_stats, col_stats, nnz_block_ptr])
    lib.cget_col_row_stats(ptrA, ptrRowStats, ptrColStats, ptrNnzrows, ct.c_float(threshold), rows, cols)
    post_call(prev_device)
    if threshold > 0.0:
        nnz_block_ptr.cumsum_(0)
    return (row_stats, col_stats, nnz_block_ptr)