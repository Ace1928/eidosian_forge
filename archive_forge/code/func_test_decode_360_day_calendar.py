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
def test_decode_360_day_calendar() -> None:
    import cftime
    calendar = '360_day'
    for year in [2010, 2011, 2012, 2013, 2014]:
        units = f'days since {year}-01-01'
        num_times = np.arange(100)
        expected = cftime.num2date(num_times, units, calendar, only_use_cftime_datetimes=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            actual = coding.times.decode_cf_datetime(num_times, units, calendar=calendar)
            assert len(w) == 0
        assert actual.dtype == np.dtype('O')
        assert_array_equal(actual, expected)