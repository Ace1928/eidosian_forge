from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_attrs_no_conflicts_compat_minimal(self):
    """make sure compat="minimal" does not silence errors"""
    ds1 = xr.Dataset({'a': ('x', [], {'a': 0})})
    ds2 = xr.Dataset({'a': ('x', [], {'a': 1})})
    with pytest.raises(xr.MergeError, match='combine_attrs'):
        xr.merge([ds1, ds2], combine_attrs='no_conflicts', compat='minimal')