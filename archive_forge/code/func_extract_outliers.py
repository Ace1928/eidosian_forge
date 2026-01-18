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
def extract_outliers(A, SA, idx):
    shapeA = SA[0]
    formatA = SA[1]
    assert formatA in ['col_turing', 'col_ampere']
    assert A.device.type == 'cuda'
    out = torch.zeros((shapeA[0], idx.numel()), dtype=torch.int8, device=A.device)
    idx_size = ct.c_int32(idx.numel())
    rows = ct.c_int32(shapeA[0])
    cols = ct.c_int32(shapeA[1])
    ptrA = get_ptr(A)
    ptrIdx = get_ptr(idx)
    ptrOut = get_ptr(out)
    prev_device = pre_call(A.device)
    if formatA == 'col_turing':
        lib.cextractOutliers_turing(ptrA, ptrIdx, ptrOut, idx_size, rows, cols)
    elif formatA == 'col_ampere':
        lib.cextractOutliers_ampere(ptrA, ptrIdx, ptrOut, idx_size, rows, cols)
    post_call(prev_device)
    return out