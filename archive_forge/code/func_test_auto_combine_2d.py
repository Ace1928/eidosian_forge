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
def test_auto_combine_2d(self):
    ds = create_test_data
    partway1 = concat([ds(0), ds(3)], dim='dim1')
    partway2 = concat([ds(1), ds(4)], dim='dim1')
    partway3 = concat([ds(2), ds(5)], dim='dim1')
    expected = concat([partway1, partway2, partway3], dim='dim2')
    datasets = [[ds(0), ds(1), ds(2)], [ds(3), ds(4), ds(5)]]
    result = combine_nested(datasets, concat_dim=['dim1', 'dim2'])
    assert_equal(result, expected)