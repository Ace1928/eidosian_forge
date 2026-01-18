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
def test_combine_by_coords_previously_failed(self):
    datasets = [Dataset({'a': ('x', [0]), 'x': [0]}), Dataset({'b': ('x', [0]), 'x': [0]}), Dataset({'a': ('x', [1]), 'x': [1]})]
    expected = Dataset({'a': ('x', [0, 1]), 'b': ('x', [0, np.nan])}, {'x': [0, 1]})
    actual = combine_by_coords(datasets)
    assert_identical(expected, actual)