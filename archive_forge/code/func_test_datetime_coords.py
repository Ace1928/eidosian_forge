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
def test_datetime_coords(self):
    ds0 = Dataset({'time': [datetime(2000, 3, 6), datetime(2001, 3, 7)]})
    ds1 = Dataset({'time': [datetime(1999, 1, 1), datetime(1999, 2, 4)]})
    expected = {(0,): ds1, (1,): ds0}
    actual, concat_dims = _infer_concat_order_from_coords([ds0, ds1])
    assert_combined_tile_ids_equal(expected, actual)
    assert concat_dims == ['time']