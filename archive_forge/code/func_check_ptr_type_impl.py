from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def check_ptr_type_impl(type_a: tl.dtype, type_b: tl.dtype, allow_ptr_a: bool) -> None:
    if type_a.is_ptr():
        if not allow_ptr_a:
            raise IncompatibleTypeErrorImpl(type_a, type_b)
        if type_b.is_ptr() and type_a != type_b:
            raise IncompatibleTypeErrorImpl(type_a, type_b)
        if type_b.is_floating():
            raise IncompatibleTypeErrorImpl(type_a, type_b)