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
def get_clean_interp_index(arr, dim: Hashable, use_coordinate: str | bool=True, strict: bool=True):
    """Return index to use for x values in interpolation or curve fitting.

    Parameters
    ----------
    arr : DataArray
        Array to interpolate or fit to a curve.
    dim : str
        Name of dimension along which to fit.
    use_coordinate : str or bool
        If use_coordinate is True, the coordinate that shares the name of the
        dimension along which interpolation is being performed will be used as the
        x values. If False, the x values are set as an equally spaced sequence.
    strict : bool
        Whether to raise errors if the index is either non-unique or non-monotonic (default).

    Returns
    -------
    Variable
        Numerical values for the x-coordinates.

    Notes
    -----
    If indexing is along the time dimension, datetime coordinates are converted
    to time deltas with respect to 1970-01-01.
    """
    from xarray.coding.cftimeindex import CFTimeIndex
    if use_coordinate is False:
        axis = arr.get_axis_num(dim)
        return np.arange(arr.shape[axis], dtype=np.float64)
    if use_coordinate is True:
        index = arr.get_index(dim)
    else:
        index = arr.coords[use_coordinate]
        if index.ndim != 1:
            raise ValueError(f'Coordinates used for interpolation must be 1D, {use_coordinate} is {index.ndim}D.')
        index = index.to_index()
    if isinstance(index, pd.MultiIndex):
        index.name = dim
    if strict:
        if not index.is_monotonic_increasing:
            raise ValueError(f'Index {index.name!r} must be monotonically increasing')
        if not index.is_unique:
            raise ValueError(f'Index {index.name!r} has duplicate values')
    if isinstance(index, (CFTimeIndex, pd.DatetimeIndex)):
        offset = type(index[0])(1970, 1, 1)
        if isinstance(index, CFTimeIndex):
            index = index.values
        index = Variable(data=datetime_to_numeric(index, offset=offset, datetime_unit='ns'), dims=(dim,))
    try:
        index = index.values.astype(np.float64)
    except (TypeError, ValueError):
        raise TypeError(f'Index {index.name!r} must be castable to float64 to support interpolation or curve fitting, got {type(index).__name__}.')
    return index