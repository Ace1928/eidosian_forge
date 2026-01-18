from __future__ import annotations
from itertools import combinations, permutations
from typing import cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import _parse_array_of_cftime_strings
from xarray.core.types import InterpOptions
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
@requires_scipy
def test_interpolate_nd_with_nan() -> None:
    """Interpolate an array with an nd indexer and `NaN` values."""
    x = [0, 1, 2]
    y = [10, 20]
    c = {'x': x, 'y': y}
    a = np.arange(6, dtype=float).reshape(2, 3)
    a[0, 1] = np.nan
    ia = xr.DataArray(a, dims=('y', 'x'), coords=c)
    da = xr.DataArray([1, 2, 2], dims='a', coords={'a': [0, 2, 4]})
    out = da.interp(a=ia)
    expected = xr.DataArray([[1.0, np.nan, 2.0], [2.0, 2.0, np.nan]], dims=('y', 'x'), coords=c)
    xr.testing.assert_allclose(out.drop_vars('a'), expected)
    db = 2 * da
    ds = xr.Dataset({'da': da, 'db': db})
    out2 = ds.interp(a=ia)
    expected_ds = xr.Dataset({'da': expected, 'db': 2 * expected})
    xr.testing.assert_allclose(out2.drop_vars('a'), expected_ds)