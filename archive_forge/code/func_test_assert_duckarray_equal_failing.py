from __future__ import annotations
import warnings
import numpy as np
import pytest
import xarray as xr
from xarray.tests import has_dask
@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('duckarray', (pytest.param(np.array, id='numpy'), pytest.param(dask_from_array, id='dask', marks=pytest.mark.skipif(not has_dask, reason='requires dask')), pytest.param(quantity, id='pint', marks=pytest.mark.skipif(not has_pint, reason='requires pint'))))
@pytest.mark.parametrize(['obj1', 'obj2'], (pytest.param([1e-10, 2], [0.0, 2.0], id='both arrays'), pytest.param([1e-17, 2], 0.0, id='second scalar'), pytest.param(0.0, [1e-17, 2], id='first scalar')))
def test_assert_duckarray_equal_failing(duckarray, obj1, obj2) -> None:
    a = duckarray(obj1)
    b = duckarray(obj2)
    with pytest.raises(AssertionError):
        xr.testing.assert_duckarray_equal(a, b)