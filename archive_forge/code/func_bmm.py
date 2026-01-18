import logging
import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Union
import torch
from xformers import _has_cpp_library, _is_triton_available
from xformers.components.attention.attention_mask import AttentionMask
def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _has_cpp_library:
        if isinstance(a, SparseCS):
            return a.spmm(b)
        if a.is_sparse:
            return _sparse_bmm(a, b)
    return a @ b