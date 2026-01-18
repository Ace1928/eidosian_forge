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
def compare_tensor_meta(a: TensorLikeType, b: TensorLikeType, check_strides=False, *, allow_rhs_unbacked=False):
    """
    Checks that two tensor likes have the same shape,
    dtype and device.

    In the future this will validate additional metadata, like
    strides.
    """
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)
    if not same_shape(a.shape, b.shape, allow_rhs_unbacked=allow_rhs_unbacked):
        msg = f'Shapes {a.shape} and {b.shape} are not equal!'
        raise AssertionError(msg)
    if a.dtype != b.dtype:
        msg = f'Dtypes {a.dtype} and {b.dtype} are not equal!'
        raise AssertionError(msg)
    if a.device != b.device:
        if (str(a.device) == 'cuda:0' or str(a.device) == 'cuda') and (str(b.device) == 'cuda:0' or str(b.device) == 'cuda'):
            pass
        else:
            msg = f'Devices {a.device} and {b.device} are not equal!'
            raise AssertionError(msg)
    if check_strides:
        same_strides, idx = check_significant_strides(a, b)
        if not same_strides:
            msg = f'Stride mismatch! Strides are {a.stride()} and {b.stride()} (mismatched at {idx})!'
            raise RuntimeError(msg)
        if a.storage_offset() != b.storage_offset():
            msg = f'Storage offset mismatch! Storage offsets are {a.storage_offset()} and {b.storage_offset()}!'
            raise RuntimeError(msg)
    if a.is_conj() != b.is_conj():
        raise RuntimeError(f'Conj mismatch! is_conj is set to {a.is_conj()} and {b.is_conj()}')
    if a.is_neg() != b.is_neg():
        raise RuntimeError(f'Neg mismatch! is_neg is set to {a.is_neg()} and {b.is_neg()}')