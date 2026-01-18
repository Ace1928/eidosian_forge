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
def masked_where(condition, a):
    cshape = getattr(condition, 'shape', ())
    if cshape and cshape != a.shape:
        raise IndexError('Inconsistent shape between the condition and the input (got %s and %s)' % (cshape, a.shape))
    condition = asanyarray(condition)
    a = asanyarray(a)
    ainds = tuple(range(a.ndim))
    cinds = tuple(range(condition.ndim))
    return blockwise(np.ma.masked_where, ainds, condition, cinds, a, ainds, dtype=a.dtype)