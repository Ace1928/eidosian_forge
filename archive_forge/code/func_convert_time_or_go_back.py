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
def convert_time_or_go_back(date, date_type):
    """Convert a single date to a new date_type (cftime.datetime or pd.Timestamp).

    If the new date is invalid, it goes back a day and tries again. If it is still
    invalid, goes back a second day.

    This is meant to convert end-of-month dates into a new calendar.
    """
    if date_type == pd.Timestamp:
        date_type = nanosecond_precision_timestamp
    try:
        return date_type(date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond)
    except OutOfBoundsDatetime:
        raise
    except ValueError:
        try:
            return date_type(date.year, date.month, date.day - 1, date.hour, date.minute, date.second, date.microsecond)
        except ValueError:
            return date_type(date.year, date.month, date.day - 2, date.hour, date.minute, date.second, date.microsecond)