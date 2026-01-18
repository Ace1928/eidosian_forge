from __future__ import annotations
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from xarray import (
from xarray.core import dtypes
from xarray.core.combine import (
from xarray.tests import assert_equal, assert_identical, requires_cftime
from xarray.tests.test_dataset import create_test_data
@requires_cftime
def test_combine_by_coords_distant_cftime_dates():
    import cftime
    time_1 = [cftime.DatetimeGregorian(4500, 12, 31)]
    time_2 = [cftime.DatetimeGregorian(4600, 12, 31)]
    time_3 = [cftime.DatetimeGregorian(5100, 12, 31)]
    da_1 = DataArray([0], dims=['time'], coords=[time_1], name='a').to_dataset()
    da_2 = DataArray([1], dims=['time'], coords=[time_2], name='a').to_dataset()
    da_3 = DataArray([2], dims=['time'], coords=[time_3], name='a').to_dataset()
    result = combine_by_coords([da_1, da_2, da_3])
    expected_time = np.concatenate([time_1, time_2, time_3])
    expected = DataArray([0, 1, 2], dims=['time'], coords=[expected_time], name='a').to_dataset()
    assert_identical(result, expected)