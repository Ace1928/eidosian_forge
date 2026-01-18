from __future__ import annotations
import contextlib
from collections.abc import Container, Iterable, Sequence
from functools import wraps
from numbers import Integral
import numpy as np
from tlz import concat
from dask.core import flatten
def argtopk_aggregate(a_plus_idx, k, axis, keepdims):
    """Final aggregation function of argtopk

    Invoke argtopk one final time, sort the results internally, drop the data
    and return the index only.
    """
    assert keepdims is True
    a_plus_idx = a_plus_idx if len(a_plus_idx) > 1 else a_plus_idx[0]
    a, idx = argtopk(a_plus_idx, k, axis, keepdims)
    axis = axis[0]
    idx2 = np.argsort(a, axis=axis)
    idx = np.take_along_axis(idx, idx2, axis)
    if k < 0:
        return idx
    return idx[tuple((slice(None, None, -1) if i == axis else slice(None) for i in range(idx.ndim)))]