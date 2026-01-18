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
def test_interp_like() -> None:
    ds = create_test_data()
    ds.attrs['foo'] = 'var'
    ds['var1'].attrs['buz'] = 'var2'
    other = xr.DataArray(np.random.randn(3), dims=['dim2'], coords={'dim2': [0, 1, 2]})
    interpolated = ds.interp_like(other)
    assert_allclose(interpolated['var1'], ds['var1'].interp(dim2=other['dim2']))
    assert_allclose(interpolated['var1'], ds['var1'].interp_like(other))
    assert interpolated['var3'].equals(ds['var3'])
    assert interpolated.attrs['foo'] == 'var'
    assert interpolated['var1'].attrs['buz'] == 'var2'
    other = xr.DataArray(np.random.randn(3), dims=['dim3'], coords={'dim3': ['a', 'b', 'c']})
    actual = ds.interp_like(other)
    expected = ds.reindex_like(other)
    assert_allclose(actual, expected)