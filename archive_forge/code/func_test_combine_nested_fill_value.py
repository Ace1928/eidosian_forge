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
@pytest.mark.parametrize('fill_value', [dtypes.NA, 2, 2.0, {'a': 2, 'b': 1}])
def test_combine_nested_fill_value(self, fill_value):
    datasets = [Dataset({'a': ('x', [2, 3]), 'b': ('x', [-2, 1]), 'x': [1, 2]}), Dataset({'a': ('x', [1, 2]), 'b': ('x', [3, -1]), 'x': [0, 1]})]
    if fill_value == dtypes.NA:
        fill_value_a = fill_value_b = np.nan
    elif isinstance(fill_value, dict):
        fill_value_a = fill_value['a']
        fill_value_b = fill_value['b']
    else:
        fill_value_a = fill_value_b = fill_value
    expected = Dataset({'a': (('t', 'x'), [[fill_value_a, 2, 3], [1, 2, fill_value_a]]), 'b': (('t', 'x'), [[fill_value_b, -2, 1], [3, -1, fill_value_b]])}, {'x': [0, 1, 2]})
    actual = combine_nested(datasets, concat_dim='t', fill_value=fill_value)
    assert_identical(expected, actual)