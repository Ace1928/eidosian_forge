from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_scipy
def test_interpolate_kwargs():
    da = xr.DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims='x')
    expected = xr.DataArray(np.array([4, 5, 6], dtype=np.float64), dims='x')
    actual = da.interpolate_na(dim='x', fill_value='extrapolate')
    assert_equal(actual, expected)
    expected = xr.DataArray(np.array([4, 5, -999], dtype=np.float64), dims='x')
    actual = da.interpolate_na(dim='x', fill_value=-999)
    assert_equal(actual, expected)