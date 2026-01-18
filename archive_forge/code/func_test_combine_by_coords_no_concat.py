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
def test_combine_by_coords_no_concat(self):
    objs = [Dataset({'x': 0}), Dataset({'y': 1})]
    actual = combine_by_coords(objs)
    expected = Dataset({'x': 0, 'y': 1})
    assert_identical(expected, actual)
    objs = [Dataset({'x': 0, 'y': 1}), Dataset({'y': np.nan, 'z': 2})]
    actual = combine_by_coords(objs)
    expected = Dataset({'x': 0, 'y': 1, 'z': 2})
    assert_identical(expected, actual)