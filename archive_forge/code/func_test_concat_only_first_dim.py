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
def test_concat_only_first_dim(self, create_combined_ids):
    shape = (2, 3)
    combined_ids = create_combined_ids(shape)
    result = _combine_all_along_first_dim(combined_ids, dim='dim1', data_vars='all', coords='different', compat='no_conflicts')
    ds = create_test_data
    partway1 = concat([ds(0), ds(3)], dim='dim1')
    partway2 = concat([ds(1), ds(4)], dim='dim1')
    partway3 = concat([ds(2), ds(5)], dim='dim1')
    expected_datasets = [partway1, partway2, partway3]
    expected = {(i,): ds for i, ds in enumerate(expected_datasets)}
    assert_combined_tile_ids_equal(result, expected)