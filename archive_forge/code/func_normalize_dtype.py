from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def normalize_dtype(dtype, parm=None):
    torch_dtype = None
    if dtype is not None:
        dtype = _dtypes.dtype(dtype)
        torch_dtype = dtype.torch_dtype
    return torch_dtype