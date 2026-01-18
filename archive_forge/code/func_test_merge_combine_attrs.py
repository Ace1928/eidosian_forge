from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
@pytest.mark.parametrize(['combine_attrs', 'attrs1', 'attrs2', 'expected_attrs', 'expect_error'], (('drop', {'a': 0, 'b': 1, 'c': 2}, {'a': 1, 'b': 2, 'c': 3}, {}, False), ('drop_conflicts', {'a': 0, 'b': 1, 'c': 2}, {'b': 2, 'c': 2, 'd': 3}, {'a': 0, 'c': 2, 'd': 3}, False), ('override', {'a': 0, 'b': 1}, {'a': 1, 'b': 2}, {'a': 0, 'b': 1}, False), ('no_conflicts', {'a': 0, 'b': 1}, {'a': 0, 'b': 2}, None, True), ('identical', {'a': 0, 'b': 1}, {'a': 0, 'b': 2}, None, True)))
def test_merge_combine_attrs(self, combine_attrs, attrs1, attrs2, expected_attrs, expect_error):
    ds1 = xr.Dataset(attrs=attrs1)
    ds2 = xr.Dataset(attrs=attrs2)
    if expect_error:
        with pytest.raises(xr.MergeError):
            ds1.merge(ds2, combine_attrs=combine_attrs)
    else:
        actual = ds1.merge(ds2, combine_attrs=combine_attrs)
        expected = xr.Dataset(attrs=expected_attrs)
        assert_identical(actual, expected)