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
def test_encode_cf_datetime_pandas_min() -> None:
    dates = pd.date_range('2000', periods=3)
    num, units, calendar = encode_cf_datetime(dates)
    expected_num = np.array([0.0, 1.0, 2.0])
    expected_units = 'days since 2000-01-01 00:00:00'
    expected_calendar = 'proleptic_gregorian'
    np.testing.assert_array_equal(num, expected_num)
    assert units == expected_units
    assert calendar == expected_calendar