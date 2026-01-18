from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def normalize_seq_array_like(x, parm=None):
    return tuple((normalize_array_like(value) for value in x))