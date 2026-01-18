from __future__ import annotations
import contextlib
import datetime
import inspect
import warnings
from functools import partial
from importlib import import_module
import numpy as np
import pandas as pd
from numpy import all as array_all  # noqa
from numpy import any as array_any  # noqa
from numpy import (  # noqa
from numpy import concatenate as _concatenate
from numpy.lib.stride_tricks import sliding_window_view  # noqa
from packaging.version import Version
from xarray.core import dask_array_ops, dtypes, nputils
from xarray.core.options import OPTIONS
from xarray.core.utils import is_duck_array, is_duck_dask_array, module_available
from xarray.namedarray import pycompat
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, is_chunked_array
def datetime_to_numeric(array, offset=None, datetime_unit=None, dtype=float):
    """Convert an array containing datetime-like data to numerical values.
    Convert the datetime array to a timedelta relative to an offset.
    Parameters
    ----------
    array : array-like
        Input data
    offset : None, datetime or cftime.datetime
        Datetime offset. If None, this is set by default to the array's minimum
        value to reduce round off errors.
    datetime_unit : {None, Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as}
        If not None, convert output to a given datetime unit. Note that some
        conversions are not allowed due to non-linear relationships between units.
    dtype : dtype
        Output dtype.
    Returns
    -------
    array
        Numerical representation of datetime object relative to an offset.
    Notes
    -----
    Some datetime unit conversions won't work, for example from days to years, even
    though some calendars would allow for them (e.g. no_leap). This is because there
    is no `cftime.timedelta` object.
    """
    if offset is None:
        if array.dtype.kind in 'Mm':
            offset = _datetime_nanmin(array)
        else:
            offset = min(array)
    if is_duck_dask_array(array) and np.issubdtype(array.dtype, object):
        array = array.map_blocks(lambda a, b: a - b, offset, meta=array._meta)
    else:
        array = array - offset
    if not hasattr(array, 'dtype'):
        array = np.array(array)
    if array.dtype.kind in 'O':
        return py_timedelta_to_float(array, datetime_unit or 'ns').astype(dtype)
    elif array.dtype.kind in 'mM':
        if datetime_unit:
            array = array / np.timedelta64(1, datetime_unit)
        return np.where(isnull(array), np.nan, array.astype(dtype))