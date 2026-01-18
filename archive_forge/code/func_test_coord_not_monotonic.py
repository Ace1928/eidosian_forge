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
def test_coord_not_monotonic(self):
    ds0 = Dataset({'x': [0, 1]})
    ds1 = Dataset({'x': [3, 2]})
    with pytest.raises(ValueError, match='Coordinate variable x is neither monotonically increasing nor'):
        _infer_concat_order_from_coords([ds1, ds0])