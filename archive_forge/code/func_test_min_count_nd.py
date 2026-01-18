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
@pytest.mark.parametrize('dtype', [float, int, np.float32, np.bool_])
@pytest.mark.parametrize('dask', [False, True])
@pytest.mark.parametrize('func', ['sum', 'prod'])
def test_min_count_nd(dtype, dask, func):
    if dask and (not has_dask):
        pytest.skip('requires dask')
    min_count = 3
    dim_num = 3
    da = construct_dataarray(dim_num, dtype, contains_nan=True, dask=dask)
    with raise_if_dask_computes():
        actual = getattr(da, func)(dim=['x', 'y', 'z'], skipna=True, min_count=min_count)
    expected = getattr(da, func)(dim=..., skipna=True, min_count=min_count)
    assert_allclose(actual, expected)
    assert_dask_array(actual, dask)