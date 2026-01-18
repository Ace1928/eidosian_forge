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
@pytest.mark.parametrize('func', ['sum', 'prod'])
def test_min_count_dataset(func):
    da = construct_dataarray(2, dtype=float, contains_nan=True, dask=False)
    ds = Dataset({'var1': da}, coords={'scalar': 0})
    actual = getattr(ds, func)(dim='x', skipna=True, min_count=3)['var1']
    expected = getattr(ds['var1'], func)(dim='x', skipna=True, min_count=3)
    assert_allclose(actual, expected)