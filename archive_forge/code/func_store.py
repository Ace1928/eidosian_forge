from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def store(pointer, value, mask=None, boundary_check=(), cache_modifier='', eviction_policy='', _builder=None):
    """
    Store a tensor of data into memory locations defined by `pointer`:
        (1) `pointer` could be a single element pointer, then a scalar will be stored

            - `mask` must be scalar too
            - `boundary_check` and `padding_option` must be empty

        (2) `pointer` could be element-wise tensor of pointers, in which case:

            - `mask` is implicitly broadcast to `pointer.shape`
            - `boundary_check` must be empty

        (3) or `pointer` could be a block pointer defined by `make_block_ptr`, in which case:

            - `mask` must be None
            - `boundary_check` can be specified to control the behavior of out-of-bound access

    `value` is implicitly broadcast to `pointer.shape` and typecast to `pointer.dtype.element_ty`.

    :param pointer: The memory location where the elements of `value` are stored
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param value: The tensor of elements to be stored
    :type value: Block
    :param mask: If `mask[idx]` is false, do not store `value[idx]` at `pointer[idx]`
    :type mask: Block of triton.int1, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    """
    value = _to_tensor(value, _builder)
    if _constexpr_to_value(mask) is not None:
        mask = _to_tensor(mask, _builder)
    cache_modifier = _constexpr_to_value(cache_modifier)
    eviction_policy = _constexpr_to_value(eviction_policy)
    return semantic.store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy, _builder)