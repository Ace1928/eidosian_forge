from __future__ import annotations
import re
import warnings
from collections.abc import Hashable
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Callable, Union
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta
from xarray.coding.variables import (
from xarray.core import indexing
from xarray.core.common import contains_cftime_datetimes, is_np_datetime_like
from xarray.core.duck_array_ops import asarray
from xarray.core.formatting import first_n_items, format_timestamp, last_item
from xarray.core.pdcompat import nanosecond_precision_timestamp
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import T_ChunkedArray, get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.namedarray.utils import is_duck_dask_array
def cftime_to_nptime(times, raise_on_invalid: bool=True) -> np.ndarray:
    """Given an array of cftime.datetime objects, return an array of
    numpy.datetime64 objects of the same size

    If raise_on_invalid is True (default), invalid dates trigger a ValueError.
    Otherwise, the invalid element is replaced by np.NaT."""
    times = np.asarray(times)
    new = np.empty(times.shape, dtype='M8[ns]')
    for i, t in np.ndenumerate(times):
        try:
            dt = nanosecond_precision_timestamp(t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond)
        except ValueError as e:
            if raise_on_invalid:
                raise ValueError(f'Cannot convert date {t} to a date in the standard calendar.  Reason: {e}.')
            else:
                dt = 'NaT'
        new[i] = np.datetime64(dt)
    return new