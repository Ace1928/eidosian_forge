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
def test_infer_order_from_coords(self):
    data = create_test_data()
    objs = [data.isel(dim2=slice(4, 9)), data.isel(dim2=slice(4))]
    actual = combine_by_coords(objs)
    expected = data
    assert expected.broadcast_equals(actual)