from __future__ import annotations
import builtins
import itertools
import operator
from typing import Optional, Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import (
from torch import broadcast_shapes
def put_along_axis(arr: ArrayLike, indices: ArrayLike, values: ArrayLike, axis):
    (arr,), axis = _util.axis_none_flatten(arr, axis=axis)
    axis = _util.normalize_axis_index(axis, arr.ndim)
    indices, values = torch.broadcast_tensors(indices, values)
    values = _util.cast_if_needed(values, arr.dtype)
    result = torch.scatter(arr, axis, indices, values)
    arr.copy_(result.reshape(arr.shape))
    return None