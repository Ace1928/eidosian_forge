from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@pytest.mark.parametrize('center', (True, False, (True, False)))
@pytest.mark.parametrize('fill_value', (np.nan, 0.0))
@pytest.mark.parametrize('dask', (True, False))
def test_ndrolling_construct(self, center, fill_value, dask) -> None:
    da = DataArray(np.arange(5 * 6 * 7).reshape(5, 6, 7).astype(float), dims=['x', 'y', 'z'], coords={'x': ['a', 'b', 'c', 'd', 'e'], 'y': np.arange(6)})
    ds = xr.Dataset({'da': da})
    if dask and has_dask:
        ds = ds.chunk({'x': 4})
    actual = ds.rolling(x=3, z=2, center=center).construct(x='x1', z='z1', fill_value=fill_value)
    if not isinstance(center, tuple):
        center = (center, center)
    expected = ds.rolling(x=3, center=center[0]).construct(x='x1', fill_value=fill_value).rolling(z=2, center=center[1]).construct(z='z1', fill_value=fill_value)
    assert_allclose(actual, expected)