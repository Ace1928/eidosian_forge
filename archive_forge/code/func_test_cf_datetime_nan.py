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
@arm_xfail
@requires_cftime
@pytest.mark.parametrize(['num_dates', 'units', 'expected_list'], [([np.nan], 'days since 2000-01-01', ['NaT']), ([np.nan, 0], 'days since 2000-01-01', ['NaT', '2000-01-01T00:00:00Z']), ([np.nan, 0, 1], 'days since 2000-01-01', ['NaT', '2000-01-01T00:00:00Z', '2000-01-02T00:00:00Z'])])
def test_cf_datetime_nan(num_dates, units, expected_list) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN')
        actual = coding.times.decode_cf_datetime(num_dates, units)
    expected = pd.to_datetime(expected_list).to_numpy(dtype='datetime64[ns]')
    assert_array_equal(expected, actual)