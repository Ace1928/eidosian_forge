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
def test_da_groupby_quantile() -> None:
    array = xr.DataArray(data=[1, 2, 3, 4, 5, 6], coords={'x': [1, 1, 1, 2, 2, 2]}, dims='x')
    expected = xr.DataArray(data=[2, 5], coords={'x': [1, 2], 'quantile': 0.5}, dims='x')
    actual = array.groupby('x').quantile(0.5)
    assert_identical(expected, actual)
    expected = xr.DataArray(data=[[1, 3], [4, 6]], coords={'x': [1, 2], 'quantile': [0, 1]}, dims=('x', 'quantile'))
    actual = array.groupby('x').quantile([0, 1])
    assert_identical(expected, actual)
    array = xr.DataArray(data=[np.nan, 2, 3, 4, 5, 6], coords={'x': [1, 1, 1, 2, 2, 2]}, dims='x')
    for skipna in (True, False, None):
        e = [np.nan, 5] if skipna is False else [2.5, 5]
        expected = xr.DataArray(data=e, coords={'x': [1, 2], 'quantile': 0.5}, dims='x')
        actual = array.groupby('x').quantile(0.5, skipna=skipna)
        assert_identical(expected, actual)
    array = xr.DataArray(data=[[1, 11, 26], [2, 12, 22], [3, 13, 23], [4, 16, 24], [5, 15, 25]], coords={'x': [1, 1, 1, 2, 2], 'y': [0, 0, 1]}, dims=('x', 'y'))
    actual_x = array.groupby('x').quantile(0, dim=...)
    expected_x = xr.DataArray(data=[1, 4], coords={'x': [1, 2], 'quantile': 0}, dims='x')
    assert_identical(expected_x, actual_x)
    actual_y = array.groupby('y').quantile(0, dim=...)
    expected_y = xr.DataArray(data=[1, 22], coords={'y': [0, 1], 'quantile': 0}, dims='y')
    assert_identical(expected_y, actual_y)
    actual_xx = array.groupby('x').quantile(0)
    expected_xx = xr.DataArray(data=[[1, 11, 22], [4, 15, 24]], coords={'x': [1, 2], 'y': [0, 0, 1], 'quantile': 0}, dims=('x', 'y'))
    assert_identical(expected_xx, actual_xx)
    actual_yy = array.groupby('y').quantile(0)
    expected_yy = xr.DataArray(data=[[1, 26], [2, 22], [3, 23], [4, 24], [5, 25]], coords={'x': [1, 1, 1, 2, 2], 'y': [0, 1], 'quantile': 0}, dims=('x', 'y'))
    assert_identical(expected_yy, actual_yy)
    times = pd.date_range('2000-01-01', periods=365)
    x = [0, 1]
    foo = xr.DataArray(np.reshape(np.arange(365 * 2), (365, 2)), coords={'time': times, 'x': x}, dims=('time', 'x'))
    g = foo.groupby(foo.time.dt.month)
    actual = g.quantile(0, dim=...)
    expected = xr.DataArray(data=[0.0, 62.0, 120.0, 182.0, 242.0, 304.0, 364.0, 426.0, 488.0, 548.0, 610.0, 670.0], coords={'month': np.arange(1, 13), 'quantile': 0}, dims='month')
    assert_identical(expected, actual)
    actual = g.quantile(0, dim='time')[:2]
    expected = xr.DataArray(data=[[0.0, 1], [62.0, 63]], coords={'month': [1, 2], 'x': [0, 1], 'quantile': 0}, dims=('month', 'x'))
    assert_identical(expected, actual)
    array = xr.DataArray(data=[1, 2, 3, 4], coords={'x': [1, 1, 2, 2]}, dims='x')
    expected = xr.DataArray(data=[1, 3], coords={'x': [1, 2], 'quantile': 0.5}, dims='x')
    actual = array.groupby('x').quantile(0.5, method='lower')
    assert_identical(expected, actual)