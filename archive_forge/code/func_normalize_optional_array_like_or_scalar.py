from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def normalize_optional_array_like_or_scalar(x, parm=None):
    if x is None:
        return None
    return normalize_array_like_or_scalar(x, parm)