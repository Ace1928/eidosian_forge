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
@pytest.mark.parametrize('dim_num', [1, 2])
@pytest.mark.parametrize('dtype', [float, int, np.float32, np.bool_])
@pytest.mark.parametrize('dask', [False, True])
@pytest.mark.parametrize('func', ['sum', 'prod'])
@pytest.mark.parametrize('aggdim', [None, 'x'])
@pytest.mark.parametrize('contains_nan', [True, False])
@pytest.mark.parametrize('skipna', [True, False, None])
def test_min_count(dim_num, dtype, dask, func, aggdim, contains_nan, skipna):
    if dask and (not has_dask):
        pytest.skip('requires dask')
    da = construct_dataarray(dim_num, dtype, contains_nan=contains_nan, dask=dask)
    min_count = 3
    with raise_if_dask_computes():
        actual = getattr(da, func)(dim=aggdim, skipna=skipna, min_count=min_count)
    expected = series_reduce(da, func, skipna=skipna, dim=aggdim, min_count=min_count)
    assert_allclose(actual, expected)
    assert_dask_array(actual, dask)