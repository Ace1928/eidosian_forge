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
def test_decode_ambiguous_time_warns(calendar) -> None:
    from cftime import num2date
    is_standard_calendar = calendar in coding.times._STANDARD_CALENDARS
    dates = [1, 2, 3]
    units = 'days since 1-1-1'
    expected = num2date(dates, units, calendar=calendar, only_use_cftime_datetimes=True)
    if is_standard_calendar:
        with pytest.warns(SerializationWarning) as record:
            result = decode_cf_datetime(dates, units, calendar=calendar)
        relevant_warnings = [r for r in record.list if str(r.message).startswith('Ambiguous reference date string: 1-1-1')]
        assert len(relevant_warnings) == 1
    else:
        with assert_no_warnings():
            result = decode_cf_datetime(dates, units, calendar=calendar)
    np.testing.assert_array_equal(result, expected)