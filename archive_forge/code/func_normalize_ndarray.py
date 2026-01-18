from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def normalize_ndarray(arg, parm=None):
    if arg is None:
        return arg
    from ._ndarray import ndarray
    if not isinstance(arg, ndarray):
        raise TypeError(f"'{parm.name}' must be an array")
    return arg.tensor