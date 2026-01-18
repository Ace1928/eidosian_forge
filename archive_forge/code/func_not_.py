from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def not_(input: tl.tensor, builder: ir.builder):
    if not input.type.is_int1():
        input = bitcast(input, tl.dtype('int1'), builder)
    return invert(input, builder)