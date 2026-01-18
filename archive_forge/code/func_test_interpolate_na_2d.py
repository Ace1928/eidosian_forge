from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_bottleneck
@pytest.mark.parametrize('coords', [pytest.param(None, marks=pytest.mark.xfail()), {'x': np.arange(4), 'y': np.arange(12)}])
def test_interpolate_na_2d(coords):
    n = np.nan
    da = xr.DataArray([[1, 2, 3, 4, n, 6, n, n, n, 10, 11, n], [n, n, 3, n, n, 6, n, n, n, 10, n, n], [n, n, 3, n, n, 6, n, n, n, 10, n, n], [n, 2, 3, 4, n, 6, n, n, n, 10, 11, n]], dims=['x', 'y'], coords=coords)
    actual = da.interpolate_na('y', max_gap=2)
    expected_y = da.copy(data=[[1, 2, 3, 4, 5, 6, n, n, n, 10, 11, n], [n, n, 3, n, n, 6, n, n, n, 10, n, n], [n, n, 3, n, n, 6, n, n, n, 10, n, n], [n, 2, 3, 4, 5, 6, n, n, n, 10, 11, n]])
    assert_equal(actual, expected_y)
    actual = da.interpolate_na('y', max_gap=1, fill_value='extrapolate')
    expected_y_extra = da.copy(data=[[1, 2, 3, 4, n, 6, n, n, n, 10, 11, 12], [n, n, 3, n, n, 6, n, n, n, 10, n, n], [n, n, 3, n, n, 6, n, n, n, 10, n, n], [1, 2, 3, 4, n, 6, n, n, n, 10, 11, 12]])
    assert_equal(actual, expected_y_extra)
    actual = da.interpolate_na('x', max_gap=3)
    expected_x = xr.DataArray([[1, 2, 3, 4, n, 6, n, n, n, 10, 11, n], [n, 2, 3, 4, n, 6, n, n, n, 10, 11, n], [n, 2, 3, 4, n, 6, n, n, n, 10, 11, n], [n, 2, 3, 4, n, 6, n, n, n, 10, 11, n]], dims=['x', 'y'], coords=coords)
    assert_equal(actual, expected_x)