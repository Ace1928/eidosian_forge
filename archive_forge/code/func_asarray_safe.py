from __future__ import annotations
import contextlib
import functools
import itertools
import math
import numbers
import warnings
import numpy as np
from tlz import concat, frequencies
from dask.array.core import Array
from dask.array.numpy_compat import AxisError
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import has_keyword, is_arraylike, is_cupy_type, typename
def asarray_safe(a, like, **kwargs):
    """
    If a is dask.array, return dask.array.asarray(a, **kwargs),
    otherwise return np.asarray(a, like=like, **kwargs), dispatching
    the call to the library that implements the like array. Note that
    when a is a dask.Array but like isn't, this function will call
    a.compute(scheduler="sync") before np.asarray, as downstream
    libraries are unlikely to know how to convert a dask.Array.
    """
    from dask.array.core import asarray
    return _array_like_safe(np.asarray, asarray, a, like, **kwargs)