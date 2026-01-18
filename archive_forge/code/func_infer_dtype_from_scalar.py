from __future__ import annotations
import datetime as dt
import functools
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import (
from pandas._libs.missing import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.compat.numpy import np_version_gt2
from pandas.errors import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import (
from pandas.io._util import _arrow_dtype_mapping
def infer_dtype_from_scalar(val) -> tuple[DtypeObj, Any]:
    """
    Interpret the dtype from a scalar.

    Parameters
    ----------
    val : object
    """
    dtype: DtypeObj = _dtype_obj
    if isinstance(val, np.ndarray):
        if val.ndim != 0:
            msg = 'invalid ndarray passed to infer_dtype_from_scalar'
            raise ValueError(msg)
        dtype = val.dtype
        val = lib.item_from_zerodim(val)
    elif isinstance(val, str):
        dtype = _dtype_obj
        if using_pyarrow_string_dtype():
            from pandas.core.arrays.string_ import StringDtype
            dtype = StringDtype(storage='pyarrow_numpy')
    elif isinstance(val, (np.datetime64, dt.datetime)):
        try:
            val = Timestamp(val)
        except OutOfBoundsDatetime:
            return (_dtype_obj, val)
        if val is NaT or val.tz is None:
            val = val.to_datetime64()
            dtype = val.dtype
        else:
            dtype = DatetimeTZDtype(unit=val.unit, tz=val.tz)
    elif isinstance(val, (np.timedelta64, dt.timedelta)):
        try:
            val = Timedelta(val)
        except (OutOfBoundsTimedelta, OverflowError):
            dtype = _dtype_obj
        else:
            if val is NaT:
                val = np.timedelta64('NaT', 'ns')
            else:
                val = val.asm8
            dtype = val.dtype
    elif is_bool(val):
        dtype = np.dtype(np.bool_)
    elif is_integer(val):
        if isinstance(val, np.integer):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.int64)
        try:
            np.array(val, dtype=dtype)
        except OverflowError:
            dtype = np.array(val).dtype
    elif is_float(val):
        if isinstance(val, np.floating):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.float64)
    elif is_complex(val):
        dtype = np.dtype(np.complex128)
    if isinstance(val, Period):
        dtype = PeriodDtype(freq=val.freq)
    elif isinstance(val, Interval):
        subtype = infer_dtype_from_scalar(val.left)[0]
        dtype = IntervalDtype(subtype=subtype, closed=val.closed)
    return (dtype, val)