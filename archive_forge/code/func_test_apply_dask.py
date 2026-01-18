from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
@requires_dask
def test_apply_dask() -> None:
    import dask.array as da
    array = da.ones((2,), chunks=2)
    variable = xr.Variable('x', array)
    coords = xr.DataArray(variable).coords.variables
    data_array = xr.DataArray(variable, dims=['x'], coords=coords)
    dataset = xr.Dataset({'y': variable})
    with pytest.raises(ValueError):
        apply_ufunc(identity, array)
    with pytest.raises(ValueError):
        apply_ufunc(identity, variable)
    with pytest.raises(ValueError):
        apply_ufunc(identity, data_array)
    with pytest.raises(ValueError):
        apply_ufunc(identity, dataset)
    with pytest.raises(ValueError):
        apply_ufunc(identity, array, dask='unknown')

    def dask_safe_identity(x):
        return apply_ufunc(identity, x, dask='allowed')
    assert array is dask_safe_identity(array)
    actual = dask_safe_identity(variable)
    assert isinstance(actual.data, da.Array)
    assert_identical(variable, actual)
    actual = dask_safe_identity(data_array)
    assert isinstance(actual.data, da.Array)
    assert_identical(data_array, actual)
    actual = dask_safe_identity(dataset)
    assert isinstance(actual['y'].data, da.Array)
    assert_identical(dataset, actual)