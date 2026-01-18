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
def test_interpolate_limits():
    da = xr.DataArray(np.array([1, 2, np.nan, np.nan, np.nan, 6], dtype=np.float64), dims='x')
    actual = da.interpolate_na(dim='x', limit=None)
    assert actual.isnull().sum() == 0
    actual = da.interpolate_na(dim='x', limit=2)
    expected = xr.DataArray(np.array([1, 2, 3, 4, np.nan, 6], dtype=np.float64), dims='x')
    assert_equal(actual, expected)