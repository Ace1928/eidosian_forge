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
def test_groupby_math_more() -> None:
    ds = create_test_data()
    grouped = ds.groupby('numbers')
    zeros = DataArray([0, 0, 0, 0], [('numbers', range(4))])
    expected = (ds + Variable('dim3', np.zeros(10))).transpose('dim3', 'dim1', 'dim2', 'time')
    actual = grouped + zeros
    assert_equal(expected, actual)
    actual = zeros + grouped
    assert_equal(expected, actual)
    with pytest.raises(ValueError, match='incompat.* grouped binary'):
        grouped + ds
    with pytest.raises(ValueError, match='incompat.* grouped binary'):
        ds + grouped
    with pytest.raises(TypeError, match='only support binary ops'):
        grouped + 1
    with pytest.raises(TypeError, match='only support binary ops'):
        grouped + grouped
    with pytest.raises(TypeError, match='in-place operations'):
        ds += grouped
    ds = Dataset({'x': ('time', np.arange(100)), 'time': pd.date_range('2000-01-01', periods=100)})
    with pytest.raises(ValueError, match='incompat.* grouped binary'):
        ds + ds.groupby('time.month')