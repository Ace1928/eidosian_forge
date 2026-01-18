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
def test_get_new_tile_ids(self, create_combined_ids):
    shape = (1, 2, 3)
    combined_ids = create_combined_ids(shape)
    expected_tile_ids = sorted(combined_ids.keys())
    actual_tile_ids = _create_tile_ids(shape)
    assert expected_tile_ids == actual_tile_ids