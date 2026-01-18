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
def test_nested_concat(self):
    objs = [Dataset({'x': [0]}), Dataset({'x': [1]})]
    expected = Dataset({'x': [0, 1]})
    actual = combine_nested(objs, concat_dim='x')
    assert_identical(expected, actual)
    actual = combine_nested(objs, concat_dim=['x'])
    assert_identical(expected, actual)
    actual = combine_nested([actual], concat_dim=None)
    assert_identical(expected, actual)
    actual = combine_nested([actual], concat_dim='x')
    assert_identical(expected, actual)
    objs = [Dataset({'x': [0, 1]}), Dataset({'x': [2]})]
    actual = combine_nested(objs, concat_dim='x')
    expected = Dataset({'x': [0, 1, 2]})
    assert_identical(expected, actual)
    objs = [Dataset({'x': ('a', [0]), 'y': ('a', [0])}), Dataset({'y': ('a', [1]), 'x': ('a', [1])})]
    actual = combine_nested(objs, concat_dim='a')
    expected = Dataset({'x': ('a', [0, 1]), 'y': ('a', [0, 1])})
    assert_identical(expected, actual)
    objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'x': [1]})]
    actual = combine_nested(objs, concat_dim='x')
    expected = Dataset({'x': [0, 1], 'y': [0]})
    assert_identical(expected, actual)