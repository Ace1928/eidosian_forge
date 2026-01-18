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
def masked_inside(x, v1, v2):
    x = asanyarray(x)
    return x.map_blocks(np.ma.masked_inside, v1, v2)