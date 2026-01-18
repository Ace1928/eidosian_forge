import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def make_triton_contiguous(t):
    """Return input as a triton-contiguous tensor.

    A triton-contiguous tensor is defined as a tensor that has strides
    with minimal value equal to 1.

    While triton kernels support triton-non-contiguous tensors (all
    strides being greater than 1 or having 0 strides) arguments, a
    considerable slow-down occurs because tensor data is copied
    element-wise rather than chunk-wise.
    """
    if min(t.stride()) != 1:
        return t.contiguous()
    else:
        return t