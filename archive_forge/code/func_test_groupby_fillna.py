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
def test_groupby_fillna(self) -> None:
    a = DataArray([np.nan, 1, np.nan, 3], coords={'x': range(4)}, dims='x')
    fill_value = DataArray([0, 1], dims='y')
    actual = a.fillna(fill_value)
    expected = DataArray([[0, 1], [1, 1], [0, 1], [3, 3]], coords={'x': range(4)}, dims=('x', 'y'))
    assert_identical(expected, actual)
    b = DataArray(range(4), coords={'x': range(4)}, dims='x')
    expected = b.copy()
    for target in [a, expected]:
        target.coords['b'] = ('x', [0, 0, 1, 1])
    actual = a.groupby('b').fillna(DataArray([0, 2], dims='b'))
    assert_identical(expected, actual)