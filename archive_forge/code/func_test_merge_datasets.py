from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_datasets(self):
    data = create_test_data(add_attrs=False)
    actual = xr.merge([data[['var1']], data[['var2']]])
    expected = data[['var1', 'var2']]
    assert_identical(actual, expected)
    actual = xr.merge([data, data])
    assert_identical(actual, data)