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
@pytest.mark.parametrize('join, expected', [('outer', Dataset({'x': [0, 1], 'y': [0, 1]})), ('inner', Dataset({'x': [0, 1], 'y': []})), ('left', Dataset({'x': [0, 1], 'y': [0]})), ('right', Dataset({'x': [0, 1], 'y': [1]}))])
def test_combine_nested_join(self, join, expected):
    objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'x': [1], 'y': [1]})]
    actual = combine_nested(objs, concat_dim='x', join=join)
    assert_identical(expected, actual)