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
def test_check_for_impossible_ordering(self):
    ds0 = Dataset({'x': [0, 1, 5]})
    ds1 = Dataset({'x': [2, 3]})
    with pytest.raises(ValueError, match='does not have monotonic global indexes along dimension x'):
        combine_by_coords([ds1, ds0])