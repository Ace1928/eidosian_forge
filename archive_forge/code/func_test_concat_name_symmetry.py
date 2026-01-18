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
def test_concat_name_symmetry(self):
    """Inspired by the discussion on GH issue #2777"""
    da1 = DataArray(name='a', data=[[0]], dims=['x', 'y'])
    da2 = DataArray(name='b', data=[[1]], dims=['x', 'y'])
    da3 = DataArray(name='a', data=[[2]], dims=['x', 'y'])
    da4 = DataArray(name='b', data=[[3]], dims=['x', 'y'])
    x_first = combine_nested([[da1, da2], [da3, da4]], concat_dim=['x', 'y'])
    y_first = combine_nested([[da1, da3], [da2, da4]], concat_dim=['y', 'x'])
    assert_identical(x_first, y_first)