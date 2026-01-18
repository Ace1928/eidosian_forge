from __future__ import annotations
from functools import partial
from operator import getitem
import numpy as np
from dask import core
from dask.array.core import Array, apply_infer_dtype, asarray, blockwise, elemwise
from dask.base import is_dask_collection, normalize_token
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from, funcname
@derived_from(np)
def frompyfunc(func, nin, nout):
    if nout > 1:
        raise NotImplementedError('frompyfunc with more than one output')
    return ufunc(da_frompyfunc(func, nin, nout))