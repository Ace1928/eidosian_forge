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
def test_groupby_cumsum() -> None:
    ds = xr.Dataset({'foo': (('x',), [7, 3, 1, 1, 1, 1, 1])}, coords={'x': [0, 1, 2, 3, 4, 5, 6], 'group_id': ('x', [0, 0, 1, 1, 2, 2, 2])})
    actual = ds.groupby('group_id').cumsum(dim='x')
    expected = xr.Dataset({'foo': (('x',), [7, 10, 1, 2, 1, 2, 3])}, coords={'x': [0, 1, 2, 3, 4, 5, 6], 'group_id': ds.group_id})
    assert_identical(expected.drop_vars(['x', 'group_id']), actual)
    actual = ds.foo.groupby('group_id').cumsum(dim='x')
    expected.coords['group_id'] = ds.group_id
    expected.coords['x'] = np.arange(7)
    assert_identical(expected.foo, actual)