from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_auto_align(self):
    ds1 = xr.Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})
    ds2 = xr.Dataset({'b': ('x', [3, 4]), 'x': [1, 2]})
    expected = xr.Dataset({'a': ('x', [1, 2, np.nan]), 'b': ('x', [np.nan, 3, 4])}, {'x': [0, 1, 2]})
    assert expected.identical(ds1.merge(ds2))
    assert expected.identical(ds2.merge(ds1))
    expected = expected.isel(x=slice(2))
    assert expected.identical(ds1.merge(ds2, join='left'))
    assert expected.identical(ds2.merge(ds1, join='right'))
    expected = expected.isel(x=slice(1, 2))
    assert expected.identical(ds1.merge(ds2, join='inner'))
    assert expected.identical(ds2.merge(ds1, join='inner'))