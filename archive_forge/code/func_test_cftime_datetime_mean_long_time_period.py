from __future__ import annotations
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy import array, nan
from xarray import DataArray, Dataset, cftime_range, concat
from xarray.core import dtypes, duck_array_ops
from xarray.core.duck_array_ops import (
from xarray.namedarray.pycompat import array_type
from xarray.testing import assert_allclose, assert_equal, assert_identical
from xarray.tests import (
@requires_cftime
def test_cftime_datetime_mean_long_time_period():
    import cftime
    times = np.array([[cftime.DatetimeNoLeap(400, 12, 31, 0, 0, 0, 0), cftime.DatetimeNoLeap(520, 12, 31, 0, 0, 0, 0)], [cftime.DatetimeNoLeap(520, 12, 31, 0, 0, 0, 0), cftime.DatetimeNoLeap(640, 12, 31, 0, 0, 0, 0)], [cftime.DatetimeNoLeap(640, 12, 31, 0, 0, 0, 0), cftime.DatetimeNoLeap(760, 12, 31, 0, 0, 0, 0)]])
    da = DataArray(times, dims=['time', 'd2'])
    result = da.mean('d2')
    expected = DataArray([cftime.DatetimeNoLeap(460, 12, 31, 0, 0, 0, 0), cftime.DatetimeNoLeap(580, 12, 31, 0, 0, 0, 0), cftime.DatetimeNoLeap(700, 12, 31, 0, 0, 0, 0)], dims=['time'])
    assert_equal(result, expected)