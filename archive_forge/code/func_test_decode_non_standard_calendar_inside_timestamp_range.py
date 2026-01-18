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
@pytest.mark.parametrize('calendar', _NON_STANDARD_CALENDARS)
def test_decode_non_standard_calendar_inside_timestamp_range(calendar) -> None:
    import cftime
    units = 'days since 0001-01-01'
    times = pd.date_range('2001-04-01-00', end='2001-04-30-23', freq='h')
    non_standard_time = cftime.date2num(times.to_pydatetime(), units, calendar=calendar)
    expected = cftime.num2date(non_standard_time, units, calendar=calendar, only_use_cftime_datetimes=True)
    expected_dtype = np.dtype('O')
    actual = coding.times.decode_cf_datetime(non_standard_time, units, calendar=calendar)
    assert actual.dtype == expected_dtype
    abs_diff = abs(actual - expected)
    assert (abs_diff <= np.timedelta64(1, 's')).all()