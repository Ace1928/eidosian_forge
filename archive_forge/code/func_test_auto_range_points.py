from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('DataFrame', DataFrames)
def test_auto_range_points(DataFrame):
    n = 10
    data = np.arange(n, dtype='i4')
    df = DataFrame({'time': np.arange(n), 'x': data, 'y': data})
    cvs = ds.Canvas(plot_width=n, plot_height=n)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((n, n), int)
    np.fill_diagonal(sol, 1)
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 9), close=True)
    assert_eq_ndarray(agg.y_range, (0, 9), close=True)
    cvs = ds.Canvas(plot_width=n + 1, plot_height=n + 1)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((n + 1, n + 1), int)
    np.fill_diagonal(sol, 1)
    sol[5, 5] = 0
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 9), close=True)
    assert_eq_ndarray(agg.y_range, (0, 9), close=True)
    n = 4
    data = np.arange(n, dtype='i4')
    df = DataFrame({'time': np.arange(n), 'x': data, 'y': data})
    cvs = ds.Canvas(plot_width=2 * n, plot_height=2 * n)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((2 * n, 2 * n), int)
    np.fill_diagonal(sol, 1)
    sol[np.array([tuple(range(1, 4, 2))])] = 0
    sol[np.array([tuple(range(4, 8, 2))])] = 0
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 3), close=True)
    assert_eq_ndarray(agg.y_range, (0, 3), close=True)
    cvs = ds.Canvas(plot_width=2 * n + 1, plot_height=2 * n + 1)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.zeros((2 * n + 1, 2 * n + 1), int)
    sol[0, 0] = 1
    sol[3, 3] = 1
    sol[6, 6] = 1
    sol[8, 8] = 1
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 3), close=True)
    assert_eq_ndarray(agg.y_range, (0, 3), close=True)