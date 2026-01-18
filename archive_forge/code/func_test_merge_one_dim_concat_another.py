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
def test_merge_one_dim_concat_another(self):
    objs = [[Dataset({'foo': ('x', [0, 1])}), Dataset({'bar': ('x', [10, 20])})], [Dataset({'foo': ('x', [2, 3])}), Dataset({'bar': ('x', [30, 40])})]]
    expected = Dataset({'foo': ('x', [0, 1, 2, 3]), 'bar': ('x', [10, 20, 30, 40])})
    actual = combine_nested(objs, concat_dim=['x', None], compat='equals')
    assert_identical(expected, actual)
    objs = [[Dataset({'foo': ('x', [0, 1])}), Dataset({'foo': ('x', [2, 3])})], [Dataset({'bar': ('x', [10, 20])}), Dataset({'bar': ('x', [30, 40])})]]
    actual = combine_nested(objs, concat_dim=[None, 'x'], compat='equals')
    assert_identical(expected, actual)