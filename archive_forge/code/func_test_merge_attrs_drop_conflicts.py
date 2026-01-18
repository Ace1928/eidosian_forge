from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_attrs_drop_conflicts(self):
    ds1 = xr.Dataset(attrs={'a': 0, 'b': 0, 'c': 0})
    ds2 = xr.Dataset(attrs={'b': 0, 'c': 1, 'd': 0})
    ds3 = xr.Dataset(attrs={'a': 0, 'b': 1, 'c': 0, 'e': 0})
    actual = xr.merge([ds1, ds2, ds3], combine_attrs='drop_conflicts')
    expected = xr.Dataset(attrs={'a': 0, 'd': 0, 'e': 0})
    assert_identical(actual, expected)