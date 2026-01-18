from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_compat_minimal(self) -> None:
    ds1 = xr.Dataset(coords={'foo': [1, 2, 3], 'bar': 4})
    ds2 = xr.Dataset(coords={'foo': [1, 2, 3], 'bar': 5})
    actual = xr.merge([ds1, ds2], compat='minimal')
    expected = xr.Dataset(coords={'foo': [1, 2, 3]})
    assert_identical(actual, expected)