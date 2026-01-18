from __future__ import annotations
import copy
import itertools
import math
import numbers
import warnings
from collections.abc import Hashable, Mapping, Sequence
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, NoReturn, cast
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import xarray as xr  # only for Dataset and DataArray
from xarray.core import common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from xarray.core.arithmetic import VariableArithmetic
from xarray.core.common import AbstractArray
from xarray.core.indexing import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import NamedArray, _raise_if_any_duplicate_dimensions
from xarray.namedarray.pycompat import integer_types, is_0d_dask_array, to_duck_array
def as_compatible_data(data: T_DuckArray | ArrayLike, fastpath: bool=False) -> T_DuckArray:
    """Prepare and wrap data to put in a Variable.

    - If data does not have the necessary attributes, convert it to ndarray.
    - If data has dtype=datetime64, ensure that it has ns precision. If it's a
      pandas.Timestamp, convert it to datetime64.
    - If data is already a pandas or xarray object (other than an Index), just
      use the values.

    Finally, wrap it up with an adapter if necessary.
    """
    if fastpath and getattr(data, 'ndim', 0) > 0:
        return cast('T_DuckArray', _maybe_wrap_data(data))
    from xarray.core.dataarray import DataArray
    if isinstance(data, (Variable, DataArray)):
        return data.data
    if isinstance(data, NON_NUMPY_SUPPORTED_ARRAY_TYPES):
        data = _possibly_convert_datetime_or_timedelta_index(data)
        return cast('T_DuckArray', _maybe_wrap_data(data))
    if isinstance(data, tuple):
        data = utils.to_0d_object_array(data)
    if isinstance(data, pd.Timestamp):
        data = np.datetime64(data.value, 'ns')
    if isinstance(data, timedelta):
        data = np.timedelta64(getattr(data, 'value', data), 'ns')
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.values
    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)
        if mask.any():
            dtype, fill_value = dtypes.maybe_promote(data.dtype)
            data = duck_array_ops.where_method(data, ~mask, fill_value)
        else:
            data = np.asarray(data)
    if not isinstance(data, np.ndarray) and (hasattr(data, '__array_function__') or hasattr(data, '__array_namespace__')):
        return cast('T_DuckArray', data)
    data = np.asarray(data)
    if isinstance(data, np.ndarray) and data.dtype.kind in 'OMm':
        data = _possibly_convert_objects(data)
    return _maybe_wrap_data(data)