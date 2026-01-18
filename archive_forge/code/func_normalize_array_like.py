from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def normalize_array_like(x, parm=None):
    from ._ndarray import asarray
    return asarray(x).tensor