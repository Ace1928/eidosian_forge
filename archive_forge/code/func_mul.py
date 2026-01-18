from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def mul(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fmul(input.handle, other.handle), input.type)
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_mul(input.handle, other.handle), input.type)
    assert False