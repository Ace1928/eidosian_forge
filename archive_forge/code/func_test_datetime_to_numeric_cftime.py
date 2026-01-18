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
@pytest.mark.parametrize('dask', [True, False])
def test_datetime_to_numeric_cftime(dask):
    if dask and (not has_dask):
        pytest.skip('requires dask')
    times = cftime_range('2000', periods=5, freq='7D', calendar='standard').values
    if dask:
        import dask.array
        times = dask.array.from_array(times, chunks=-1)
    with raise_if_dask_computes():
        result = duck_array_ops.datetime_to_numeric(times, datetime_unit='h', dtype=int)
    expected = 24 * np.arange(0, 35, 7)
    np.testing.assert_array_equal(result, expected)
    offset = times[1]
    with raise_if_dask_computes():
        result = duck_array_ops.datetime_to_numeric(times, offset=offset, datetime_unit='h', dtype=int)
    expected = 24 * np.arange(-7, 28, 7)
    np.testing.assert_array_equal(result, expected)
    dtype = np.float32
    with raise_if_dask_computes():
        result = duck_array_ops.datetime_to_numeric(times, datetime_unit='h', dtype=dtype)
    expected = 24 * np.arange(0, 35, 7).astype(dtype)
    np.testing.assert_array_equal(result, expected)
    with raise_if_dask_computes():
        if dask:
            time = dask.array.asarray(times[1])
        else:
            time = np.asarray(times[1])
        result = duck_array_ops.datetime_to_numeric(time, offset=times[0], datetime_unit='h', dtype=int)
    expected = np.array(24 * 7).astype(int)
    np.testing.assert_array_equal(result, expected)