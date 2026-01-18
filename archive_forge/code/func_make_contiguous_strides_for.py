from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def make_contiguous_strides_for(shape: ShapeType, row_major: bool=True) -> Tuple[int, ...]:
    """
    Returns the strides of a contiguous tensor if row_major
    If row_major=True, it returns the strides of a contiguous batch of Fortran-contiguous matrices
    This is often used when calling external libraries like BLAS/LAPACK/cuSolver...
    """
    validate_shape(shape)
    if not shape:
        return ()

    def _is_singleton(s):
        if not isinstance(s, torch.SymInt):
            return False
        if s.node.singleton_int() is not None:
            return True
        return s.node.is_symbolic() and s.node.hint is not None and isinstance(s.node.hint, torch.SymInt) and (s.node.hint.node.singleton_int() is not None)
    multiplier = 1
    strides = []
    for l in reversed(shape):
        strides.append(multiplier)
        multiplier *= l if _is_singleton(l) else sym_max(l, 1)
    result = tuple(reversed(strides))
    if row_major:
        return result
    else:
        if len(shape) < 2:
            return result
        return result[:-2] + (1, max(shape[-2], 1))