from __future__ import annotations
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy import array, nan
from xarray import DataArray, Dataset, cftime_range, concat
from xarray.core import dtypes, duck_array_ops
from xarray.core.duck_array_ops import (
from xarray.namedarray.pycompat import array_type
from xarray.testing import assert_allclose, assert_equal, assert_identical
from xarray.tests import (
@requires_cftime
@pytest.mark.parametrize('dask', [False, True])
def test_cftime_datetime_mean(dask):
    if dask and (not has_dask):
        pytest.skip('requires dask')
    times = cftime_range('2000', periods=4)
    da = DataArray(times, dims=['time'])
    da_2d = DataArray(times.values.reshape(2, 2))
    if dask:
        da = da.chunk({'time': 2})
        da_2d = da_2d.chunk({'dim_0': 2})
    expected = da.isel(time=0)
    with raise_if_dask_computes(max_computes=1):
        result = da.isel(time=0).mean()
    assert_dask_array(result, dask)
    assert_equal(result, expected)
    expected = DataArray(times.date_type(2000, 1, 2, 12))
    with raise_if_dask_computes(max_computes=1):
        result = da.mean()
    assert_dask_array(result, dask)
    assert_equal(result, expected)
    with raise_if_dask_computes(max_computes=1):
        result = da_2d.mean()
    assert_dask_array(result, dask)
    assert_equal(result, expected)