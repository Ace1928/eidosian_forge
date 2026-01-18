from __future__ import annotations
import datetime as dt
import warnings
from collections.abc import Hashable, Sequence
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, get_args
import numpy as np
import pandas as pd
from xarray.core import utils
from xarray.core.common import _contains_datetime_like_objects, ones_like
from xarray.core.computation import apply_ufunc
from xarray.core.duck_array_ops import (
from xarray.core.options import _get_keep_attrs
from xarray.core.types import Interp1dOptions, InterpOptions
from xarray.core.utils import OrderedSet, is_scalar
from xarray.core.variable import Variable, broadcast_variables
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def interp_func(var, x, new_x, method: InterpOptions, kwargs):
    """
    multi-dimensional interpolation for array-like. Interpolated axes should be
    located in the last position.

    Parameters
    ----------
    var : np.ndarray or dask.array.Array
        Array to be interpolated. The final dimension is interpolated.
    x : a list of 1d array.
        Original coordinates. Should not contain NaN.
    new_x : a list of 1d array
        New coordinates. Should not contain NaN.
    method : string
        {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'} for
        1-dimensional interpolation.
        {'linear', 'nearest'} for multidimensional interpolation
    **kwargs
        Optional keyword arguments to be passed to scipy.interpolator

    Returns
    -------
    interpolated: array
        Interpolated array

    Notes
    -----
    This requires scipy installed.

    See Also
    --------
    scipy.interpolate.interp1d
    """
    if not x:
        return var.copy()
    if len(x) == 1:
        func, kwargs = _get_interpolator(method, vectorizeable_only=True, **kwargs)
    else:
        func, kwargs = _get_interpolator_nd(method, **kwargs)
    if is_chunked_array(var):
        chunkmanager = get_chunked_array_type(var)
        ndim = var.ndim
        nconst = ndim - len(x)
        out_ind = list(range(nconst)) + list(range(ndim, ndim + new_x[0].ndim))
        x_arginds = [[_x, (nconst + index,)] for index, _x in enumerate(x)]
        x_arginds = [item for pair in x_arginds for item in pair]
        new_x_arginds = [[_x, [ndim + index for index in range(_x.ndim)]] for _x in new_x]
        new_x_arginds = [item for pair in new_x_arginds for item in pair]
        args = (var, range(ndim), *x_arginds, *new_x_arginds)
        _, rechunked = chunkmanager.unify_chunks(*args)
        args = tuple((elem for pair in zip(rechunked, args[1::2]) for elem in pair))
        new_x = rechunked[1 + (len(rechunked) - 1) // 2:]
        new_x0_chunks = new_x[0].chunks
        new_x0_shape = new_x[0].shape
        new_x0_chunks_is_not_none = new_x0_chunks is not None
        new_axes = {ndim + i: new_x0_chunks[i] if new_x0_chunks_is_not_none else new_x0_shape[i] for i in range(new_x[0].ndim)}
        localize = method in ['linear', 'nearest'] and new_x0_chunks_is_not_none
        if not issubclass(var.dtype.type, np.inexact):
            dtype = float
        else:
            dtype = var.dtype
        meta = var._meta
        return chunkmanager.blockwise(_chunked_aware_interpnd, out_ind, *args, interp_func=func, interp_kwargs=kwargs, localize=localize, concatenate=True, dtype=dtype, new_axes=new_axes, meta=meta, align_arrays=False)
    return _interpnd(var, x, new_x, func, kwargs)