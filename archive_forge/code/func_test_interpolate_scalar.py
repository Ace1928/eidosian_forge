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
@pytest.mark.parametrize('method', ['linear'])
@pytest.mark.parametrize('case', [pytest.param(0, id='no_chunk'), pytest.param(1, id='chunk_y')])
def test_interpolate_scalar(method: InterpOptions, case: int) -> None:
    if not has_scipy:
        pytest.skip('scipy is not installed.')
    if not has_dask and case in [1]:
        pytest.skip('dask is not installed in the environment.')
    da = get_example_data(case)
    xdest = 0.4
    actual = da.interp(x=xdest, method=method)

    def func(obj, new_x):
        return scipy.interpolate.interp1d(da['x'], obj.data, axis=obj.get_axis_num('x'), bounds_error=False, fill_value=np.nan)(new_x)
    coords = {'x': xdest, 'y': da['y'], 'x2': func(da['x2'], xdest)}
    expected = xr.DataArray(func(da, xdest), dims=['y'], coords=coords)
    assert_allclose(actual, expected)