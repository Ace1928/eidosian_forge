from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def load(pointer, mask=None, other=None, boundary_check=tuple(), padding_option='', cache_modifier='', eviction_policy='', volatile=False, _builder=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:
        (1) `pointer` could be a single element pointer, then a scalar will be loaded

            - `mask` and `other` must be scalar too
            - `other` is implicitly typecast to `pointer.dtype.element_ty`
            - `boundary_check` and `padding_option` must be empty

        (2) `pointer` could be element-wise tensor of pointers, in which case:

            - `mask` and `other` are implicitly broadcast to `pointer.shape`
            - `other` is implicitly typecast to `pointer.dtype.element_ty`
            - `boundary_check` and `padding_option` must be empty

        (3) `pointer` could be a block pointer defined by `make_block_ptr`, in which case:

            - `mask` and `other` must be None
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access

    :param pointer: Pointer to the data to be loaded
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`
        (must be `None` with block pointers)
    :type mask: Block of `triton.int1`, optional
    :param other: if `mask[idx]` is false, return `other[idx]`
    :type other: Block, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param padding_option: should be one of {"", "zero", "nan"}, do padding while out of bound
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    """
    if _constexpr_to_value(mask) is not None:
        mask = _to_tensor(mask, _builder)
    if _constexpr_to_value(other) is not None:
        other = _to_tensor(other, _builder)
    padding_option = _constexpr_to_value(padding_option)
    cache_modifier = _constexpr_to_value(cache_modifier)
    eviction_policy = _constexpr_to_value(eviction_policy)
    volatile = _constexpr_to_value(volatile)
    return semantic.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy, volatile, _builder)