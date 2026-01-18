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
def coerce_depth_type(ndim, depth):
    for i in range(ndim):
        if isinstance(depth[i], tuple):
            depth[i] = tuple((int(d) for d in depth[i]))
        else:
            depth[i] = int(depth[i])
    return depth