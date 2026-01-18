from __future__ import annotations
import warnings
from numbers import Integral, Number
import numpy as np
from tlz import concat, get, partial
from tlz.curried import map
from dask.array import chunk
from dask.array.core import Array, concatenate, map_blocks, unify_chunks
from dask.array.creation import empty_like, full_like
from dask.array.numpy_compat import normalize_axis_tuple
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayOverlapLayer
from dask.utils import derived_from
def periodic(x, axis, depth):
    """Copy a slice of an array around to its other side

    Useful to create periodic boundary conditions for overlap
    """
    left = (slice(None, None, None),) * axis + (slice(0, depth),) + (slice(None, None, None),) * (x.ndim - axis - 1)
    right = (slice(None, None, None),) * axis + (slice(-depth, None),) + (slice(None, None, None),) * (x.ndim - axis - 1)
    l = x[left]
    r = x[right]
    l, r = _remove_overlap_boundaries(l, r, axis, depth)
    return concatenate([r, x, l], axis=axis)