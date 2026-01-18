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
def test_decode_cf_datetime_non_iso_strings() -> None:
    expected = pd.date_range(periods=100, start='2000-01-01', freq='h')
    cases = [(np.arange(100), 'hours since 2000-01-01 0'), (np.arange(100), 'hours since 2000-1-1 0'), (np.arange(100), 'hours since 2000-01-01 0:00')]
    for num_dates, units in cases:
        actual = coding.times.decode_cf_datetime(num_dates, units)
        abs_diff = abs(actual - expected.values)
        assert (abs_diff <= np.timedelta64(1, 's')).all()