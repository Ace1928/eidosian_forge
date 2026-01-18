from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_allclose, assert_array_equal, mock
from xarray.tests import assert_identical as assert_identical_
def test_gufunc_methods():
    xarray_obj = xr.DataArray([1, 2, 3])
    with pytest.raises(NotImplementedError, match='reduce method'):
        np.add.reduce(xarray_obj, 1)