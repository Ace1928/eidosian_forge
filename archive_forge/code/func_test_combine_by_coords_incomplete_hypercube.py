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
def test_combine_by_coords_incomplete_hypercube(self):
    x1 = Dataset({'a': (('y', 'x'), [[1]])}, coords={'y': [0], 'x': [0]})
    x2 = Dataset({'a': (('y', 'x'), [[1]])}, coords={'y': [1], 'x': [0]})
    x3 = Dataset({'a': (('y', 'x'), [[1]])}, coords={'y': [0], 'x': [1]})
    actual = combine_by_coords([x1, x2, x3])
    expected = Dataset({'a': (('y', 'x'), [[1, 1], [1, np.nan]])}, coords={'y': [0, 1], 'x': [0, 1]})
    assert_identical(expected, actual)
    with pytest.raises(ValueError):
        combine_by_coords([x1, x2, x3], fill_value=None)