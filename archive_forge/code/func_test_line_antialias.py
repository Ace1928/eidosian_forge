from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_line_antialias():
    x_range = y_range = (-0.1875, 1.1875)
    cvs = ds.Canvas(plot_width=11, plot_height=11, x_range=x_range, y_range=y_range)
    kwargs = dict(source=line_antialias_df, x='x0', y='y0', line_width=1)
    agg = cvs.line(agg=ds.any(), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_0, close=True)
    agg = cvs.line(agg=ds.count(self_intersect=False), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_0, close=True)
    agg = cvs.line(agg=ds.count(self_intersect=True), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_0_intersect, close=True)
    agg = cvs.line(agg=ds.sum('value', self_intersect=False), **kwargs)
    assert_eq_ndarray(agg.data, 3 * line_antialias_sol_0, close=True)
    agg = cvs.line(agg=ds.sum('value', self_intersect=True), **kwargs)
    assert_eq_ndarray(agg.data, 3 * line_antialias_sol_0_intersect, close=True)
    agg = cvs.line(agg=ds.max('value'), **kwargs)
    assert_eq_ndarray(agg.data, 3 * line_antialias_sol_0, close=True)
    agg = cvs.line(agg=ds.min('value'), **kwargs)
    assert_eq_ndarray(agg.data, 3 * line_antialias_sol_0, close=True)
    agg = cvs.line(agg=ds.first('value'), **kwargs)
    assert_eq_ndarray(agg.data, 3 * line_antialias_sol_0, close=True)
    agg = cvs.line(agg=ds.last('value'), **kwargs)
    assert_eq_ndarray(agg.data, 3 * line_antialias_sol_0, close=True)
    agg = cvs.line(agg=ds._count_ignore_antialiasing('value'), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_count_ignore_aa_0, close=True)
    agg = cvs.line(agg=ds.mean('value'), **kwargs)
    sol = 3 * line_antialias_sol_0_intersect / line_antialias_sol_count_ignore_aa_0
    assert_eq_ndarray(agg.data, sol, close=True)
    agg = cvs.line(agg=ds._min_row_index(), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_min_index_0)
    agg = cvs.line(agg=ds._max_row_index(), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_max_index_0)
    agg = cvs.line(agg=ds._min_n_row_index(n=2), **kwargs)
    assert_eq_ndarray(agg[:, :, 0].data, line_antialias_sol_min_index_0)
    sol = np.full((11, 11), -1)
    sol[(4, 5, 5, 5, 6, 1, 2), (5, 4, 5, 6, 5, 9, 9)] = 2
    sol[8:10, 9] = 1
    assert_eq_ndarray(agg[:, :, 1].data, sol)
    agg = cvs.line(agg=ds._max_n_row_index(n=2), **kwargs)
    assert_eq_ndarray(agg[:, :, 0].data, line_antialias_sol_max_index_0)
    sol = np.full((11, 11), -1)
    sol[(4, 5, 5, 5, 6, 8, 9), (5, 4, 5, 6, 5, 9, 9)] = 0
    sol[1:3, 9] = 1
    assert_eq_ndarray(agg[:, :, 1].data, sol)
    agg = cvs.line(agg=ds.max_n('value', n=2), **kwargs)
    assert_eq_ndarray(agg[:, :, 0].data, 3 * line_antialias_sol_0, close=True)
    sol = np.full((11, 11), np.nan)
    sol[(1, 5, 9), (9, 5, 9)] = 3.0
    sol[(2, 4, 5, 5, 6, 8), (9, 5, 4, 6, 5, 9)] = 0.87868
    assert_eq_ndarray(agg[:, :, 1].data, sol, close=True)
    agg = cvs.line(agg=ds.min_n('value', n=2), **kwargs)
    sol = 3 * line_antialias_sol_0
    sol[(2, 8), (9, 9)] = 0.87868
    assert_eq_ndarray(agg[:, :, 0].data, sol, close=True)
    sol = np.full((11, 11), np.nan)
    sol[(1, 2, 5, 8, 9), (9, 9, 5, 9, 9)] = 3.0
    sol[(4, 5, 5, 6), (5, 4, 6, 5)] = 0.87868
    assert_eq_ndarray(agg[:, :, 1].data, sol, close=True)
    agg = cvs.line(agg=ds.first_n('value', n=2), **kwargs)
    sol = 3 * line_antialias_sol_0
    sol[8, 9] = 0.87868
    assert_eq_ndarray(agg[:, :, 0].data, sol, close=True)
    sol = np.full((11, 11), np.nan)
    sol[(1, 5, 8, 9), (9, 5, 9, 9)] = 3.0
    sol[(2, 4, 5, 5, 6), (9, 5, 4, 6, 5)] = 0.87868
    assert_eq_ndarray(agg[:, :, 1].data, sol, close=True)
    kwargs = dict(source=line_antialias_df, x='x1', y='y1', line_width=1)
    agg = cvs.line(agg=ds.any(), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_1, close=True)
    agg = cvs.line(agg=ds.count(self_intersect=False), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_1, close=True)
    agg = cvs.line(agg=ds.count(self_intersect=True), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_1, close=True)
    agg = cvs.line(agg=ds.sum('value', self_intersect=False), **kwargs)
    assert_eq_ndarray(agg.data, 3 * line_antialias_sol_1, close=True)
    agg = cvs.line(agg=ds.sum('value', self_intersect=True), **kwargs)
    assert_eq_ndarray(agg.data, 3 * line_antialias_sol_1, close=True)
    agg = cvs.line(agg=ds.max('value'), **kwargs)
    assert_eq_ndarray(agg.data, 3 * line_antialias_sol_1, close=True)
    agg = cvs.line(agg=ds.min('value'), **kwargs)
    assert_eq_ndarray(agg.data, 3 * line_antialias_sol_1, close=True)
    agg = cvs.line(agg=ds.first('value'), **kwargs)
    assert_eq_ndarray(agg.data, 3 * line_antialias_sol_1, close=True)
    agg = cvs.line(agg=ds.last('value'), **kwargs)
    assert_eq_ndarray(agg.data, 3 * line_antialias_sol_1, close=True)
    agg = cvs.line(agg=ds._count_ignore_antialiasing('value'), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_count_ignore_aa_1, close=True)
    agg = cvs.line(agg=ds.mean('value'), **kwargs)
    sol = 3 * line_antialias_sol_1 / line_antialias_sol_count_ignore_aa_1
    assert_eq_ndarray(agg.data, sol, close=True)
    agg = cvs.line(agg=ds._min_row_index(), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_min_index_1)
    agg = cvs.line(agg=ds._max_row_index(), **kwargs)
    assert_eq_ndarray(agg.data, line_antialias_sol_max_index_1)
    agg = cvs.line(agg=ds._min_n_row_index(n=2), **kwargs)
    assert_eq_ndarray(agg[:, :, 0].data, line_antialias_sol_min_index_1)
    sol = np.full((11, 11), -1)
    sol[(2, 2, 3), (3, 4, 4)] = 1
    sol[2:4, 6:8] = 2
    assert_eq_ndarray(agg[:, :, 1].data, sol)
    agg = cvs.line(agg=ds._max_n_row_index(n=2), **kwargs)
    assert_eq_ndarray(agg[:, :, 0].data, line_antialias_sol_max_index_1)
    sol = np.full((11, 11), -1)
    sol[(2, 2, 3), (3, 4, 4)] = 0
    sol[2:4, 6:8] = 1
    assert_eq_ndarray(agg[:, :, 1].data, sol)
    agg = cvs.line(agg=ds.max_n('value', n=2), **kwargs)
    assert_eq_ndarray(agg[:, :, 0].data, 3 * line_antialias_sol_1, close=True)
    sol = np.full((11, 11), np.nan)
    sol[2, 3:8] = (0.911939, 1.83381, nan, 1.43795, 0.667619)
    sol[(3, 3, 3), (4, 6, 7)] = (0.4, 0.940874, 0.309275)
    assert_eq_ndarray(agg[:, :, 1].data, sol, close=True)
    agg = cvs.line(agg=ds.min_n('value', n=2), **kwargs)
    sol = np.full((11, 11), np.nan)
    sol[2, 1:-1] = [3.0, 2.77563, 0.911939, 1.83381, 2.1025216, 1.43795, 0.667619, 1.429411, 1.205041]
    sol[3, 1:-1] = [0.008402, 0.232772, 0.457142, 0.4, 0.905881, 0.940874, 0.309275, 1.578991, 1.8]
    assert_eq_ndarray(agg[:, :, 0].data, sol, close=True)
    sol = np.full((11, 11), np.nan)
    sol[2, 3:8] = (2.55126, 2.32689, nan, 1.878151, 1.653781)
    sol[(3, 3, 3), (4, 6, 7)] = (0.681512, 1.130251, 1.354621)
    assert_eq_ndarray(agg[:, :, 1].data, sol, close=True)
    agg = cvs.line(agg=ds.first_n('value', n=2), **kwargs)
    sol = 3 * line_antialias_sol_1
    sol[(2, 2, 3, 3), (4, 7, 4, 7)] = (1.83381, 0.667619, 0.4, 0.309275)
    assert_eq_ndarray(agg[:, :, 0].data, sol, close=True)
    sol = np.full((11, 11), np.nan)
    sol[2, 3:8] = (0.911939, 2.32689, nan, 1.43795, 1.653781)
    sol[(3, 3, 3), (4, 6, 7)] = (0.681512, 0.940874, 1.354621)
    assert_eq_ndarray(agg[:, :, 1].data, sol, close=True)
    agg = cvs.line(agg=ds.last_n('value', n=2), **kwargs)
    sol = 3 * line_antialias_sol_1
    sol[(2, 2, 3), (3, 6, 6)] = (0.911939, 1.43795, 0.940874)
    assert_eq_ndarray(agg[:, :, 0].data, sol, close=True)
    kwargs = dict(source=line_antialias_df, x=['x0', 'x1'], y=['y0', 'y1'], line_width=1)
    agg = cvs.line(agg=ds.any(), **kwargs)
    sol_max = nanmax(line_antialias_sol_0, line_antialias_sol_1)
    assert_eq_ndarray(agg.data, sol_max, close=True)
    agg = cvs.line(agg=ds.count(self_intersect=False), **kwargs)
    sol_count = nansum(line_antialias_sol_0, line_antialias_sol_1)
    assert_eq_ndarray(agg.data, sol_count, close=True)
    agg = cvs.line(agg=ds.count(self_intersect=True), **kwargs)
    sol_count_intersect = nansum(line_antialias_sol_0_intersect, line_antialias_sol_1)
    assert_eq_ndarray(agg.data, sol_count_intersect, close=True)
    agg = cvs.line(agg=ds.sum('value', self_intersect=False), **kwargs)
    assert_eq_ndarray(agg.data, 3 * sol_count, close=True)
    agg = cvs.line(agg=ds.sum('value', self_intersect=True), **kwargs)
    assert_eq_ndarray(agg.data, 3 * sol_count_intersect, close=True)
    agg = cvs.line(agg=ds.max('value'), **kwargs)
    assert_eq_ndarray(agg.data, 3 * sol_max, close=True)
    agg = cvs.line(agg=ds.min('value'), **kwargs)
    sol_min = nanmin(line_antialias_sol_0, line_antialias_sol_1)
    assert_eq_ndarray(agg.data, 3 * sol_min, close=True)
    agg = cvs.line(agg=ds.first('value'), **kwargs)
    sol_first = 3 * np.where(np.isnan(line_antialias_sol_0), line_antialias_sol_1, line_antialias_sol_0)
    assert_eq_ndarray(agg.data, sol_first, close=True)
    agg = cvs.line(agg=ds.last('value'), **kwargs)
    sol_last = 3 * np.where(np.isnan(line_antialias_sol_1), line_antialias_sol_0, line_antialias_sol_1)
    assert_eq_ndarray(agg.data, sol_last, close=True)
    agg = cvs.line(agg=ds._count_ignore_antialiasing('value'), **kwargs)
    sol = line_antialias_sol_count_ignore_aa_0 + line_antialias_sol_count_ignore_aa_1
    assert_eq_ndarray(agg.data, sol, close=True)
    agg = cvs.line(agg=ds.mean('value'), **kwargs)
    numerator = np.nan_to_num(line_antialias_sol_0_intersect) + np.nan_to_num(line_antialias_sol_1)
    denom = np.nan_to_num(line_antialias_sol_count_ignore_aa_0) + np.nan_to_num(line_antialias_sol_count_ignore_aa_1)
    with np.errstate(invalid='ignore'):
        sol = 3 * numerator / denom
    assert_eq_ndarray(agg.data, sol, close=True)
    agg = cvs.line(agg=ds._min_row_index(), **kwargs)
    sol_min_row = rowmin(line_antialias_sol_min_index_0, line_antialias_sol_min_index_1)
    assert_eq_ndarray(agg.data, sol_min_row)
    agg = cvs.line(agg=ds._max_row_index(), **kwargs)
    sol_max_row = rowmax(line_antialias_sol_max_index_0, line_antialias_sol_max_index_1)
    assert_eq_ndarray(agg.data, sol_max_row)
    agg = cvs.line(agg=ds._min_n_row_index(n=2), **kwargs)
    assert_eq_ndarray(agg.data[:, :, 0], sol_min_row)
    agg = cvs.line(agg=ds._max_n_row_index(n=2), **kwargs)
    assert_eq_ndarray(agg.data[:, :, 0], sol_max_row)
    assert_eq_ndarray(agg.x_range, x_range, close=True)
    assert_eq_ndarray(agg.y_range, y_range, close=True)