from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def array_or_scalar(values, py_type=float, return_scalar=False):
    if return_scalar:
        return py_type(values.item())
    else:
        from ._ndarray import ndarray
        return ndarray(values)