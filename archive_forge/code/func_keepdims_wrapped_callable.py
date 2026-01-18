from __future__ import annotations
import contextlib
from collections.abc import Container, Iterable, Sequence
from functools import wraps
from numbers import Integral
import numpy as np
from tlz import concat
from dask.core import flatten
@wraps(a_callable)
def keepdims_wrapped_callable(x, axis=None, keepdims=None, *args, **kwargs):
    r = a_callable(x, *args, axis=axis, **kwargs)
    if not keepdims:
        return r
    axes = axis
    if axes is None:
        axes = range(x.ndim)
    if not isinstance(axes, (Container, Iterable, Sequence)):
        axes = [axes]
    r_slice = tuple()
    for each_axis in range(x.ndim):
        if each_axis in axes:
            r_slice += (None,)
        else:
            r_slice += (slice(None),)
    r = r[r_slice]
    return r