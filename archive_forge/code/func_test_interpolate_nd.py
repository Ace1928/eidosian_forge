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
@pytest.mark.parametrize('case', [pytest.param(3, id='no_chunk'), pytest.param(4, id='chunked')])
def test_interpolate_nd(case: int) -> None:
    if not has_scipy:
        pytest.skip('scipy is not installed.')
    if not has_dask and case == 4:
        pytest.skip('dask is not installed in the environment.')
    da = get_example_data(case)
    xdestnp = np.linspace(0.1, 1.0, 11)
    ydestnp = np.linspace(0.0, 0.2, 10)
    actual = da.interp(x=xdestnp, y=ydestnp, method='linear')
    expected = da.interp(x=xdestnp, method='linear')
    expected = expected.interp(y=ydestnp, method='linear')
    assert_allclose(actual.transpose('x', 'y', 'z'), expected.transpose('x', 'y', 'z'))
    xdest = xr.DataArray(np.linspace(0.1, 1.0, 11), dims='y')
    ydest = xr.DataArray(np.linspace(0.0, 0.2, 11), dims='y')
    actual = da.interp(x=xdest, y=ydest, method='linear')
    expected_data = scipy.interpolate.RegularGridInterpolator((da['x'], da['y']), da.transpose('x', 'y', 'z').values, method='linear', bounds_error=False, fill_value=np.nan)(np.stack([xdest, ydest], axis=-1))
    expected = xr.DataArray(expected_data, dims=['y', 'z'], coords={'z': da['z'], 'y': ydest, 'x': ('y', xdest.values), 'x2': da['x2'].interp(x=xdest)})
    assert_allclose(actual.transpose('y', 'z'), expected)
    actual = da.interp(y=ydest, x=xdest, method='linear')
    assert_allclose(actual.transpose('y', 'z'), expected)