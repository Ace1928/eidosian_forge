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
def maybe_infer_to_datetimelike(value: npt.NDArray[np.object_]) -> np.ndarray | DatetimeArray | TimedeltaArray | PeriodArray | IntervalArray:
    """
    we might have a array (or single object) that is datetime like,
    and no dtype is passed don't change the value unless we find a
    datetime/timedelta set

    this is pretty strict in that a datetime/timedelta is REQUIRED
    in addition to possible nulls/string likes

    Parameters
    ----------
    value : np.ndarray[object]

    Returns
    -------
    np.ndarray, DatetimeArray, TimedeltaArray, PeriodArray, or IntervalArray

    """
    if not isinstance(value, np.ndarray) or value.dtype != object:
        raise TypeError(type(value))
    if value.ndim != 1:
        raise ValueError(value.ndim)
    if not len(value):
        return value
    return lib.maybe_convert_objects(value, convert_numeric=False, convert_non_numeric=True, dtype_if_all_nat=np.dtype('M8[ns]'))