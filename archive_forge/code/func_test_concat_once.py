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
@pytest.mark.parametrize('concat_dim', ['dim1', 'new_dim'])
def test_concat_once(self, create_combined_ids, concat_dim):
    shape = (2,)
    combined_ids = create_combined_ids(shape)
    ds = create_test_data
    result = _combine_all_along_first_dim(combined_ids, dim=concat_dim, data_vars='all', coords='different', compat='no_conflicts')
    expected_ds = concat([ds(0), ds(1)], dim=concat_dim)
    assert_combined_tile_ids_equal(result, {(): expected_ds})