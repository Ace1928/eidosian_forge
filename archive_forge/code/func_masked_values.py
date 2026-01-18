from __future__ import annotations
from functools import wraps
import numpy as np
from dask.array import chunk
from dask.array.core import asanyarray, blockwise, elemwise, map_blocks
from dask.array.reductions import reduction
from dask.array.routines import _average
from dask.array.routines import nonzero as _nonzero
from dask.base import normalize_token
from dask.utils import derived_from
@derived_from(np.ma)
def masked_values(x, value, rtol=1e-05, atol=1e-08, shrink=True):
    x = asanyarray(x)
    if getattr(value, 'shape', ()):
        raise ValueError("da.ma.masked_values doesn't support array `value`s")
    return map_blocks(np.ma.masked_values, x, value, rtol=rtol, atol=atol, shrink=shrink)