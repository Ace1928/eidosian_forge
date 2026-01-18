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
@normalize_token.register(np.ma.masked_array)
def normalize_masked_array(x):
    data = normalize_token(x.data)
    mask = normalize_token(x.mask)
    fill_value = normalize_token(x.fill_value)
    return (data, mask, fill_value)