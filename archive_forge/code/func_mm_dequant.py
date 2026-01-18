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
def mm_dequant(A, quant_state, row_stats, col_stats, out=None, new_row_stats=None, new_col_stats=None, bias=None):
    assert A.dtype == torch.int32
    if bias is not None:
        assert bias.dtype == torch.float16
    out_shape = quant_state[0]
    if len(out_shape) == 3:
        out_shape = (out_shape[0] * out_shape[1], out_shape[2])
    if out is None:
        out = torch.empty(out_shape, dtype=torch.float16, device=A.device)
    if new_row_stats is None:
        new_row_stats = torch.empty(out_shape[0], dtype=torch.float32, device=A.device)
    if new_col_stats is None:
        new_col_stats = torch.empty(out_shape[1], dtype=torch.float32, device=A.device)
    assert new_row_stats.shape[0] == row_stats.shape[0], f'{new_row_stats.shape} vs {row_stats.shape}'
    assert new_col_stats.shape[0] == col_stats.shape[0], f'{new_col_stats.shape} vs {col_stats.shape}'
    prev_device = pre_call(A.device)
    ptrA = get_ptr(A)
    ptrOut = get_ptr(out)
    ptrRowStats = get_ptr(row_stats)
    ptrColStats = get_ptr(col_stats)
    ptrNewRowStats = get_ptr(new_row_stats)
    ptrNewColStats = get_ptr(new_col_stats)
    ptrBias = get_ptr(bias)
    numRows = ct.c_int32(out_shape[0])
    numCols = ct.c_int32(out_shape[1])
    is_on_gpu([A, row_stats, col_stats, out, new_row_stats, new_col_stats, bias])
    lib.cdequant_mm_int32_fp16(ptrA, ptrRowStats, ptrColStats, ptrOut, ptrNewRowStats, ptrNewColStats, ptrBias, numRows, numCols)
    post_call(prev_device)
    return out