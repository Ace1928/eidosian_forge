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
@pytest.mark.parametrize('method', ['cubic', 'zero'])
def test_interpolate_1d_methods(method: InterpOptions) -> None:
    if not has_scipy:
        pytest.skip('scipy is not installed.')
    da = get_example_data(0)
    dim = 'x'
    xdest = np.linspace(0.0, 0.9, 80)
    actual = da.interp(method=method, coords={dim: xdest})

    def func(obj, new_x):
        return scipy.interpolate.interp1d(da[dim], obj.data, axis=obj.get_axis_num(dim), bounds_error=False, fill_value=np.nan, kind=method)(new_x)
    coords = {'x': xdest, 'y': da['y'], 'x2': ('x', func(da['x2'], xdest))}
    expected = xr.DataArray(func(da, xdest), dims=['x', 'y'], coords=coords)
    assert_allclose(actual, expected)