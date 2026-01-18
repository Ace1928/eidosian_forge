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
@pytest.mark.parametrize('old_id, new_id', [((3, 0, 1), (0, 1)), ((0, 0), (0,)), ((1,), ()), ((0,), ()), ((1, 0), (0,))])
def test_new_tile_id(self, old_id, new_id):
    ds = create_test_data
    assert _new_tile_id((old_id, ds)) == new_id