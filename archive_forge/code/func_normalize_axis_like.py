from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def normalize_axis_like(arg, parm=None):
    from ._ndarray import ndarray
    if isinstance(arg, ndarray):
        arg = operator.index(arg)
    return arg