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
def test_ignore_empty_list(self):
    ds = create_test_data(0)
    input = [ds, []]
    expected = {(0,): ds}
    actual = _infer_concat_order_from_positions(input)
    assert_combined_tile_ids_equal(expected, actual)