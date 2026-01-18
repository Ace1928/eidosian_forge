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
def set_fill_value(a, fill_value):
    a = asanyarray(a)
    if getattr(fill_value, 'shape', ()):
        raise ValueError("da.ma.set_fill_value doesn't support array `value`s")
    fill_value = np.ma.core._check_fill_value(fill_value, a.dtype)
    res = a.map_blocks(_set_fill_value, fill_value)
    a.dask = res.dask
    a._name = res.name