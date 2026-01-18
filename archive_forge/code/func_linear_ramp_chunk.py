from __future__ import annotations
import itertools
from collections.abc import Sequence
from functools import partial
from itertools import product
from numbers import Integral, Number
import numpy as np
from tlz import sliding_window
from dask.array import chunk
from dask.array.backends import array_creation_dispatch
from dask.array.core import (
from dask.array.numpy_compat import AxisError
from dask.array.ufunc import greater_equal, rint
from dask.array.utils import meta_from_array
from dask.array.wrap import empty, full, ones, zeros
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import cached_cumsum, derived_from, is_cupy_type
def linear_ramp_chunk(start, stop, num, dim, step):
    """
    Helper function to find the linear ramp for a chunk.
    """
    num1 = num + 1
    shape = list(start.shape)
    shape[dim] = num
    shape = tuple(shape)
    dtype = np.dtype(start.dtype)
    result = np.empty_like(start, shape=shape, dtype=dtype)
    for i in np.ndindex(start.shape):
        j = list(i)
        j[dim] = slice(None)
        j = tuple(j)
        result[j] = np.linspace(start[i], stop, num1, dtype=dtype)[1:][::step]
    return result