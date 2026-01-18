from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def invert(input: tl.tensor, builder: tl.tensor) -> tl.tensor:
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_ptr() or input_sca_ty.is_floating():
        raise ValueError('wrong type argument to unary invert (' + input_sca_ty.__repr__() + ')')
    _1 = tl.tensor(builder.get_all_ones_value(input_sca_ty.to_ir(builder)), input_sca_ty)
    return xor_(input, _1, builder)