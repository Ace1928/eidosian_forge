from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def minus(input: tl.tensor, builder: ir.builder) -> tl.tensor:
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_ptr():
        raise ValueError('wrong type argument to unary minus (' + input_sca_ty.__repr__() + ')')
    _0 = tl.tensor(builder.get_null_value(input_sca_ty.to_ir(builder)), input_sca_ty)
    return sub(_0, input, builder)