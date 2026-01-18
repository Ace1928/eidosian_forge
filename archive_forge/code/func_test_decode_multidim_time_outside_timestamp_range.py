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
@pytest.mark.parametrize('calendar', _ALL_CALENDARS)
def test_decode_multidim_time_outside_timestamp_range(calendar) -> None:
    from datetime import datetime
    import cftime
    units = 'days since 0001-01-01'
    times1 = [datetime(1, 4, day) for day in range(1, 6)]
    times2 = [datetime(1, 5, day) for day in range(1, 6)]
    time1 = cftime.date2num(times1, units, calendar=calendar)
    time2 = cftime.date2num(times2, units, calendar=calendar)
    mdim_time = np.empty((len(time1), 2))
    mdim_time[:, 0] = time1
    mdim_time[:, 1] = time2
    expected1 = cftime.num2date(time1, units, calendar, only_use_cftime_datetimes=True)
    expected2 = cftime.num2date(time2, units, calendar, only_use_cftime_datetimes=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Unable to decode time axis')
        actual = coding.times.decode_cf_datetime(mdim_time, units, calendar=calendar)
    assert actual.dtype == np.dtype('O')
    abs_diff1 = abs(actual[:, 0] - expected1)
    abs_diff2 = abs(actual[:, 1] - expected2)
    assert (abs_diff1 <= np.timedelta64(1, 's')).all()
    assert (abs_diff2 <= np.timedelta64(1, 's')).all()