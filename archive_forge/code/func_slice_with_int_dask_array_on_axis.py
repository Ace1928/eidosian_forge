from __future__ import annotations
import bisect
import functools
import math
import warnings
from itertools import product
from numbers import Integral, Number
from operator import itemgetter
import numpy as np
from tlz import concat, memoize, merge, pluck
from dask import config, core, utils
from dask.array.chunk import getitem
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import cached_cumsum, is_arraylike
def slice_with_int_dask_array_on_axis(x, idx, axis):
    """Slice a ND dask array with a 1D dask arrays of ints along the given
    axis.

    This is a helper function of :func:`slice_with_int_dask_array`.
    """
    from dask.array import chunk
    from dask.array.core import Array, blockwise, from_array
    from dask.array.utils import asarray_safe
    assert 0 <= axis < x.ndim
    if np.isnan(x.chunks[axis]).any():
        raise NotImplementedError('Slicing an array with unknown chunks with a dask.array of ints is not supported')
    offset = np.roll(np.cumsum(asarray_safe(x.chunks[axis], like=x._meta)), 1)
    offset[0] = 0
    offset = from_array(offset, chunks=1)
    offset = Array(offset.dask, offset.name, (x.chunks[axis],), offset.dtype, meta=x._meta)
    x_axes = tuple(range(x.ndim))
    idx_axes = (x.ndim,)
    offset_axes = (axis,)
    p_axes = x_axes[:axis + 1] + idx_axes + x_axes[axis + 1:]
    y_axes = x_axes[:axis] + idx_axes + x_axes[axis + 1:]
    p = blockwise(chunk.slice_with_int_dask_array, p_axes, x, x_axes, idx, idx_axes, offset, offset_axes, x_size=x.shape[axis], axis=axis, dtype=x.dtype, meta=x._meta)
    y = blockwise(chunk.slice_with_int_dask_array_aggregate, y_axes, idx, idx_axes, p, p_axes, concatenate=True, x_chunks=x.chunks[axis], axis=axis, dtype=x.dtype, meta=x._meta)
    return y