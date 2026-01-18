from __future__ import annotations
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime
from xarray import (
from xarray.coding.times import (
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.core.utils import is_duck_dask_array
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
@requires_cftime
def test_decode_cf_datetime_overflow() -> None:
    from cftime import DatetimeGregorian
    datetime = DatetimeGregorian
    units = 'days since 2000-01-01 00:00:00'
    days = (-117608, 95795)
    expected = (datetime(1677, 12, 31), datetime(2262, 4, 12))
    for i, day in enumerate(days):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Unable to decode time axis')
            result = coding.times.decode_cf_datetime(day, units)
        assert result == expected[i]