from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_compat(self):
    ds1 = xr.Dataset({'x': 0})
    ds2 = xr.Dataset({'x': 1})
    for compat in ['broadcast_equals', 'equals', 'identical', 'no_conflicts']:
        with pytest.raises(xr.MergeError):
            ds1.merge(ds2, compat=compat)
    ds2 = xr.Dataset({'x': [0, 0]})
    for compat in ['equals', 'identical']:
        with pytest.raises(ValueError, match='should be coordinates or not'):
            ds1.merge(ds2, compat=compat)
    ds2 = xr.Dataset({'x': ((), 0, {'foo': 'bar'})})
    with pytest.raises(xr.MergeError):
        ds1.merge(ds2, compat='identical')
    with pytest.raises(ValueError, match='compat=.* invalid'):
        ds1.merge(ds2, compat='foobar')
    assert ds1.identical(ds1.merge(ds2, compat='override'))