from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
def test_interpolate():
    vals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    expected = xr.DataArray(vals, dims='x')
    mvals = vals.copy()
    mvals[2] = np.nan
    missing = xr.DataArray(mvals, dims='x')
    actual = missing.interpolate_na(dim='x')
    assert_equal(actual, expected)