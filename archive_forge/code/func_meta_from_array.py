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
def meta_from_array(x, ndim=None, dtype=None):
    """Normalize an array to appropriate meta object

    Parameters
    ----------
    x: array-like, callable
        Either an object that looks sufficiently like a Numpy array,
        or a callable that accepts shape and dtype keywords
    ndim: int
        Number of dimensions of the array
    dtype: Numpy dtype
        A valid input for ``np.dtype``

    Returns
    -------
    array-like with zero elements of the correct dtype
    """
    if hasattr(x, '_meta') and is_dask_collection(x) and is_arraylike(x):
        x = x._meta
    if dtype is None and x is None:
        raise ValueError('You must specify the meta or dtype of the array')
    if np.isscalar(x):
        x = np.array(x)
    if x is None:
        x = np.ndarray
    elif dtype is None and hasattr(x, 'dtype'):
        dtype = x.dtype
    if isinstance(x, type):
        x = x(shape=(0,) * (ndim or 0), dtype=dtype)
    if isinstance(x, list) or isinstance(x, tuple):
        ndims = [0 if isinstance(a, numbers.Number) else a.ndim if hasattr(a, 'ndim') else len(a) for a in x]
        a = [a if nd == 0 else meta_from_array(a, nd) for a, nd in zip(x, ndims)]
        return a if isinstance(x, list) else tuple(x)
    if not hasattr(x, 'shape') or not hasattr(x, 'dtype') or (not isinstance(x.shape, tuple)):
        return x
    if ndim is None:
        ndim = x.ndim
    try:
        meta = x[tuple((slice(0, 0, None) for _ in range(x.ndim)))]
        if meta.ndim != ndim:
            if ndim > x.ndim:
                meta = meta[(Ellipsis,) + tuple((None for _ in range(ndim - meta.ndim)))]
                meta = meta[tuple((slice(0, 0, None) for _ in range(meta.ndim)))]
            elif ndim == 0:
                meta = meta.sum()
            else:
                meta = meta.reshape((0,) * ndim)
        if meta is np.ma.masked:
            meta = np.ma.array(np.empty((0,) * ndim, dtype=dtype or x.dtype), mask=True)
    except Exception:
        meta = np.empty((0,) * ndim, dtype=dtype or x.dtype)
    if np.isscalar(meta):
        meta = np.array(meta)
    if dtype and meta.dtype != dtype:
        try:
            meta = meta.astype(dtype)
        except ValueError as e:
            if any((s in str(e) for s in ['invalid literal', 'could not convert string to float'])) and meta.dtype.kind in 'SU':
                meta = np.array([]).astype(dtype)
            else:
                raise e
    return meta