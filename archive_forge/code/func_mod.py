from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def mod(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if scalar_ty.is_floating():
        ret = sub(input, mul(floor(fdiv(input, other, False, builder), builder), other, builder), builder)
        return ret
    elif scalar_ty.is_int():
        if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
            raise ValueError('Cannot mod ' + scalar_ty.__repr__() + ' by ' + other_scalar_ty.__repr__() + ' because they have different signedness;this is unlikely to result in a useful answer. Cast them to the same signedness.')
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_srem(input.handle, other.handle), input.type)
        else:
            return tl.tensor(builder.create_urem(input.handle, other.handle), input.type)
    assert False