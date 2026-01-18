from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_no_conflicts_broadcast(self):
    datasets = [xr.Dataset({'x': ('y', [0])}), xr.Dataset({'x': np.nan})]
    actual = xr.merge(datasets)
    expected = xr.Dataset({'x': ('y', [0])})
    assert_identical(expected, actual)
    datasets = [xr.Dataset({'x': ('y', [np.nan])}), xr.Dataset({'x': 0})]
    actual = xr.merge(datasets)
    assert_identical(expected, actual)