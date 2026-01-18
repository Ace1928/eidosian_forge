from __future__ import annotations
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from xarray import (
from xarray.core import dtypes
from xarray.core.combine import (
from xarray.tests import assert_equal, assert_identical, requires_cftime
from xarray.tests.test_dataset import create_test_data
def test_nested_concat_along_new_dim(self):
    objs = [Dataset({'a': ('x', [10]), 'x': [0]}), Dataset({'a': ('x', [20]), 'x': [0]})]
    expected = Dataset({'a': (('t', 'x'), [[10], [20]]), 'x': [0]})
    actual = combine_nested(objs, concat_dim='t')
    assert_identical(expected, actual)
    dim = DataArray([100, 150], name='baz', dims='baz')
    expected = Dataset({'a': (('baz', 'x'), [[10], [20]]), 'x': [0], 'baz': [100, 150]})
    actual = combine_nested(objs, concat_dim=dim)
    assert_identical(expected, actual)