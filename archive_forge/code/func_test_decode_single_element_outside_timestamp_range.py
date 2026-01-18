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
def test_decode_single_element_outside_timestamp_range(calendar) -> None:
    import cftime
    units = 'days since 0001-01-01'
    for days in [1, 1470376]:
        for num_time in [days, [days], [[days]]]:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Unable to decode time axis')
                actual = coding.times.decode_cf_datetime(num_time, units, calendar=calendar)
            expected = cftime.num2date(days, units, calendar, only_use_cftime_datetimes=True)
            assert isinstance(actual.item(), type(expected))