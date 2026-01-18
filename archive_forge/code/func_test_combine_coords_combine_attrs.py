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
@pytest.mark.parametrize('combine_attrs, expected', [('drop', Dataset({'x': [0, 1], 'y': [0, 1]}, attrs={})), ('no_conflicts', Dataset({'x': [0, 1], 'y': [0, 1]}, attrs={'a': 1, 'b': 2})), ('override', Dataset({'x': [0, 1], 'y': [0, 1]}, attrs={'a': 1})), (lambda attrs, context: attrs[1], Dataset({'x': [0, 1], 'y': [0, 1]}, attrs={'a': 1, 'b': 2}))])
def test_combine_coords_combine_attrs(self, combine_attrs, expected):
    objs = [Dataset({'x': [0], 'y': [0]}, attrs={'a': 1}), Dataset({'x': [1], 'y': [1]}, attrs={'a': 1, 'b': 2})]
    actual = combine_nested(objs, concat_dim='x', join='outer', combine_attrs=combine_attrs)
    assert_identical(expected, actual)
    if combine_attrs == 'no_conflicts':
        objs[1].attrs['a'] = 2
        with pytest.raises(ValueError, match="combine_attrs='no_conflicts'"):
            actual = combine_nested(objs, concat_dim='x', join='outer', combine_attrs=combine_attrs)