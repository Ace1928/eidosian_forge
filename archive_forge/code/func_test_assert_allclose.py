from __future__ import annotations
import warnings
import numpy as np
import pytest
import xarray as xr
from xarray.tests import has_dask
@pytest.mark.parametrize('obj1,obj2', (pytest.param(xr.Variable('x', [1e-17, 2]), xr.Variable('x', [0, 3]), id='Variable'), pytest.param(xr.DataArray([1e-17, 2], dims='x'), xr.DataArray([0, 3], dims='x'), id='DataArray'), pytest.param(xr.Dataset({'a': ('x', [1e-17, 2]), 'b': ('y', [-2e-18, 2])}), xr.Dataset({'a': ('x', [0, 2]), 'b': ('y', [0, 1])}), id='Dataset')))
def test_assert_allclose(obj1, obj2) -> None:
    with pytest.raises(AssertionError):
        xr.testing.assert_allclose(obj1, obj2)