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
def interp_na(self, dim: Hashable | None=None, use_coordinate: bool | str=True, method: InterpOptions='linear', limit: int | None=None, max_gap: int | float | str | pd.Timedelta | np.timedelta64 | dt.timedelta=None, keep_attrs: bool | None=None, **kwargs):
    """Interpolate values according to different methods."""
    from xarray.coding.cftimeindex import CFTimeIndex
    if dim is None:
        raise NotImplementedError('dim is a required argument')
    if limit is not None:
        valids = _get_valid_fill_mask(self, dim, limit)
    if max_gap is not None:
        max_type = type(max_gap).__name__
        if not is_scalar(max_gap):
            raise ValueError('max_gap must be a scalar.')
        if dim in self._indexes and isinstance(self._indexes[dim].to_pandas_index(), (pd.DatetimeIndex, CFTimeIndex)) and use_coordinate:
            max_gap = timedelta_to_numeric(max_gap)
        if not use_coordinate:
            if not isinstance(max_gap, (Number, np.number)):
                raise TypeError(f'Expected integer or floating point max_gap since use_coordinate=False. Received {max_type}.')
    index = get_clean_interp_index(self, dim, use_coordinate=use_coordinate)
    interp_class, kwargs = _get_interpolator(method, **kwargs)
    interpolator = partial(func_interpolate_na, interp_class, **kwargs)
    if keep_attrs is None:
        keep_attrs = _get_keep_attrs(default=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'overflow', RuntimeWarning)
        warnings.filterwarnings('ignore', 'invalid value', RuntimeWarning)
        arr = apply_ufunc(interpolator, self, index, input_core_dims=[[dim], [dim]], output_core_dims=[[dim]], output_dtypes=[self.dtype], dask='parallelized', vectorize=True, keep_attrs=keep_attrs).transpose(*self.dims)
    if limit is not None:
        arr = arr.where(valids)
    if max_gap is not None:
        if dim not in self.coords:
            raise NotImplementedError('max_gap not implemented for unlabeled coordinates yet.')
        nan_block_lengths = _get_nan_block_lengths(self, dim, index)
        arr = arr.where(nan_block_lengths <= max_gap)
    return arr