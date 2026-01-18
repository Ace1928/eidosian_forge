from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_allclose, assert_array_equal, mock
from xarray.tests import assert_identical as assert_identical_
def test_gufuncs():
    xarray_obj = xr.DataArray([1, 2, 3])
    fake_gufunc = mock.Mock(signature='(n)->()', autospec=np.sin)
    with pytest.raises(NotImplementedError, match='generalized ufuncs'):
        xarray_obj.__array_ufunc__(fake_gufunc, '__call__', xarray_obj)