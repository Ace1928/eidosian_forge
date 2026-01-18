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
def test_check_depths(self):
    ds = create_test_data(0)
    combined_tile_ids = {(0,): ds, (0, 1): ds}
    with pytest.raises(ValueError, match='sub-lists do not have consistent depths'):
        _check_shape_tile_ids(combined_tile_ids)