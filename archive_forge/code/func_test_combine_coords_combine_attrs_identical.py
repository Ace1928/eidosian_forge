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
def test_combine_coords_combine_attrs_identical(self):
    objs = [Dataset({'x': [0], 'y': [0]}, attrs={'a': 1}), Dataset({'x': [1], 'y': [1]}, attrs={'a': 1})]
    expected = Dataset({'x': [0, 1], 'y': [0, 1]}, attrs={'a': 1})
    actual = combine_nested(objs, concat_dim='x', join='outer', combine_attrs='identical')
    assert_identical(expected, actual)
    objs[1].attrs['b'] = 2
    with pytest.raises(ValueError, match="combine_attrs='identical'"):
        actual = combine_nested(objs, concat_dim='x', join='outer', combine_attrs='identical')