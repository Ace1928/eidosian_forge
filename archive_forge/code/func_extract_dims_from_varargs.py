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
def extract_dims_from_varargs(dims: Union[DimsSequenceType, Tuple[DimsSequenceType, ...]]) -> DimsSequenceType:
    if dims and isinstance(dims[0], Sequence):
        assert len(dims) == 1
        dims = cast(Tuple[DimsSequenceType], dims)
        return dims[0]
    else:
        return cast(DimsSequenceType, dims)