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
def test_combine_nested_missing_data_new_dim(self):
    datasets = [Dataset({'a': ('x', [2, 3]), 'x': [1, 2]}), Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})]
    expected = Dataset({'a': (('t', 'x'), [[np.nan, 2, 3], [1, 2, np.nan]])}, {'x': [0, 1, 2]})
    actual = combine_nested(datasets, concat_dim='t')
    assert_identical(expected, actual)