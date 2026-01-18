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
def test_2d_plus_bystander_dim(self):
    ds0 = Dataset({'x': [0, 1], 'y': [10, 20, 30], 't': [0.1, 0.2]})
    ds1 = Dataset({'x': [2, 3], 'y': [10, 20, 30], 't': [0.1, 0.2]})
    ds2 = Dataset({'x': [0, 1], 'y': [40, 50, 60], 't': [0.1, 0.2]})
    ds3 = Dataset({'x': [2, 3], 'y': [40, 50, 60], 't': [0.1, 0.2]})
    expected = {(0, 0): ds0, (1, 0): ds1, (0, 1): ds2, (1, 1): ds3}
    actual, concat_dims = _infer_concat_order_from_coords([ds1, ds0, ds3, ds2])
    assert_combined_tile_ids_equal(expected, actual)
    assert concat_dims == ['x', 'y']