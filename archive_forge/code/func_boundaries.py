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
def boundaries(x, depth=None, kind=None):
    """Add boundary conditions to an array before overlapping

    See Also
    --------
    periodic
    constant
    """
    if not isinstance(kind, dict):
        kind = {i: kind for i in range(x.ndim)}
    if not isinstance(depth, dict):
        depth = {i: depth for i in range(x.ndim)}
    for i in range(x.ndim):
        d = depth.get(i, 0)
        if d == 0:
            continue
        this_kind = kind.get(i, 'none')
        if this_kind == 'none':
            continue
        elif this_kind == 'periodic':
            x = periodic(x, i, d)
        elif this_kind == 'reflect':
            x = reflect(x, i, d)
        elif this_kind == 'nearest':
            x = nearest(x, i, d)
        elif i in kind:
            x = constant(x, i, d, kind[i])
    return x