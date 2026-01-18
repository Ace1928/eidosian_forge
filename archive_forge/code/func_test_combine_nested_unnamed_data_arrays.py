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
def test_combine_nested_unnamed_data_arrays(self):
    unnamed_array = DataArray(data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
    actual = combine_nested([unnamed_array], concat_dim='x')
    expected = unnamed_array
    assert_identical(expected, actual)
    unnamed_array1 = DataArray(data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
    unnamed_array2 = DataArray(data=[3.0, 4.0], coords={'x': [2, 3]}, dims='x')
    actual = combine_nested([unnamed_array1, unnamed_array2], concat_dim='x')
    expected = DataArray(data=[1.0, 2.0, 3.0, 4.0], coords={'x': [0, 1, 2, 3]}, dims='x')
    assert_identical(expected, actual)
    da1 = DataArray(data=[[0.0]], coords={'x': [0], 'y': [0]}, dims=['x', 'y'])
    da2 = DataArray(data=[[1.0]], coords={'x': [0], 'y': [1]}, dims=['x', 'y'])
    da3 = DataArray(data=[[2.0]], coords={'x': [1], 'y': [0]}, dims=['x', 'y'])
    da4 = DataArray(data=[[3.0]], coords={'x': [1], 'y': [1]}, dims=['x', 'y'])
    objs = [[da1, da2], [da3, da4]]
    expected = DataArray(data=[[0.0, 1.0], [2.0, 3.0]], coords={'x': [0, 1], 'y': [0, 1]}, dims=['x', 'y'])
    actual = combine_nested(objs, concat_dim=['x', 'y'])
    assert_identical(expected, actual)