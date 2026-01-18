from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
def test_groupby_dataset_squeeze_None() -> None:
    """Delete when removing squeeze."""
    data = Dataset({'z': (['x', 'y'], np.random.randn(3, 5))}, {'x': ('x', list('abc')), 'c': ('x', [0, 1, 0]), 'y': range(5)})
    groupby = data.groupby('x')
    assert len(groupby) == 3
    expected_groups = {'a': 0, 'b': 1, 'c': 2}
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        assert groupby.groups == expected_groups
    expected_items = [('a', data.isel(x=0)), ('b', data.isel(x=1)), ('c', data.isel(x=2))]
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        for actual1, expected1 in zip(groupby, expected_items):
            assert actual1[0] == expected1[0]
            assert_equal(actual1[1], expected1[1])

    def identity(x):
        return x
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        for k in ['x', 'c']:
            actual2 = data.groupby(k).map(identity)
            assert_equal(data, actual2)