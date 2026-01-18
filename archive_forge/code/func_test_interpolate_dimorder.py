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
@pytest.mark.parametrize('case', [pytest.param(0, id='2D'), pytest.param(3, id='3D')])
def test_interpolate_dimorder(case: int) -> None:
    """Make sure the resultant dimension order is consistent with .sel()"""
    if not has_scipy:
        pytest.skip('scipy is not installed.')
    da = get_example_data(case)
    new_x = xr.DataArray([0, 1, 2], dims='x')
    assert da.interp(x=new_x).dims == da.sel(x=new_x, method='nearest').dims
    new_y = xr.DataArray([0, 1, 2], dims='y')
    actual = da.interp(x=new_x, y=new_y).dims
    expected = da.sel(x=new_x, y=new_y, method='nearest').dims
    assert actual == expected
    actual = da.interp(y=new_y, x=new_x).dims
    expected = da.sel(y=new_y, x=new_x, method='nearest').dims
    assert actual == expected
    new_x = xr.DataArray([0, 1, 2], dims='a')
    assert da.interp(x=new_x).dims == da.sel(x=new_x, method='nearest').dims
    assert da.interp(y=new_x).dims == da.sel(y=new_x, method='nearest').dims
    new_y = xr.DataArray([0, 1, 2], dims='a')
    actual = da.interp(x=new_x, y=new_y).dims
    expected = da.sel(x=new_x, y=new_y, method='nearest').dims
    assert actual == expected
    new_x = xr.DataArray([[0], [1], [2]], dims=['a', 'b'])
    assert da.interp(x=new_x).dims == da.sel(x=new_x, method='nearest').dims
    assert da.interp(y=new_x).dims == da.sel(y=new_x, method='nearest').dims
    if case == 3:
        new_x = xr.DataArray([[0], [1], [2]], dims=['a', 'b'])
        new_z = xr.DataArray([[0], [1], [2]], dims=['a', 'b'])
        actual = da.interp(x=new_x, z=new_z).dims
        expected = da.sel(x=new_x, z=new_z, method='nearest').dims
        assert actual == expected
        actual = da.interp(z=new_z, x=new_x).dims
        expected = da.sel(z=new_z, x=new_x, method='nearest').dims
        assert actual == expected
        actual = da.interp(x=0.5, z=new_z).dims
        expected = da.sel(x=0.5, z=new_z, method='nearest').dims
        assert actual == expected