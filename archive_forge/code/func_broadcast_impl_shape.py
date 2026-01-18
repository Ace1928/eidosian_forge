from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def broadcast_impl_shape(input: tl.tensor, shape: List[int], builder: ir.builder) -> tl.tensor:
    if not input.type.is_block():
        ret_ty = tl.block_type(input.type, shape)
        return tl.tensor(builder.create_splat(input.handle, shape), ret_ty)
    src_shape = input.type.get_block_shapes()
    if len(src_shape) != len(shape):
        raise ValueError(f'Cannot broadcast, rank mismatch: {src_shape}, {shape}')
    if shape == src_shape:
        return input
    for i, item in enumerate(src_shape):
        if shape[i] != item and item != 1:
            raise ValueError(f'Cannot broadcast, the expanded size of the tensor ({shape[i]}) must match the existing size ({item}) at non-singleton dimension {i}: {src_shape}, {shape}')
    ret_ty = tl.block_type(input.type.scalar, shape)
    return tl.tensor(builder.create_broadcast(input.handle, shape), ret_ty)