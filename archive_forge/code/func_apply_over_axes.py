from __future__ import annotations
import math
import warnings
from collections.abc import Iterable
from functools import partial, reduce, wraps
from numbers import Integral, Real
import numpy as np
from tlz import concat, interleave, sliding_window
from dask.array import chunk
from dask.array.core import (
from dask.array.creation import arange, diag, empty, indices, tri
from dask.array.einsumfuncs import einsum  # noqa
from dask.array.numpy_compat import NUMPY_GE_200
from dask.array.reductions import reduction
from dask.array.ufunc import multiply, sqrt
from dask.array.utils import (
from dask.array.wrap import ones
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.delayed import Delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.utils import apply, derived_from, funcname, is_arraylike, is_cupy_type
@derived_from(np)
def apply_over_axes(func, a, axes):
    a = asarray(a)
    try:
        axes = tuple(axes)
    except TypeError:
        axes = (axes,)
    sl = a.ndim * (slice(None),)
    result = a
    for i in axes:
        result = apply_along_axis(func, i, result, 0)
        if result.ndim == a.ndim - 1:
            result = result[sl[:i] + (None,)]
        elif result.ndim != a.ndim:
            raise ValueError('func must either preserve dimensionality of the input or reduce it by one.')
    return result