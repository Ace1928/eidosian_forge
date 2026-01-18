from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.core import duck_array_ops
from xarray.tests import (
@pytest.mark.parametrize('da', (1, 2), indirect=True)
@pytest.mark.parametrize('window', (1, 2, 3, 4))
@pytest.mark.parametrize('name', ('sum', 'mean', 'std', 'max'))
def test_coarsen_da_reduce(da, window, name) -> None:
    if da.isnull().sum() > 1 and window == 1:
        pytest.skip('These parameters lead to all-NaN slices')
    coarsen_obj = da.coarsen(time=window, boundary='trim')
    actual = coarsen_obj.reduce(getattr(np, f'nan{name}'))
    expected = getattr(coarsen_obj, name)()
    assert_allclose(actual, expected)