from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_line_antialias_where():
    df = pd.DataFrame(dict(y0=[0.5, 1.0, 0.0], y1=[1.0, 0.0, 0.5], y2=[0.0, 0.5, 1.0], value=[2.2, 3.3, 1.1], other=[-9.0, -7.0, -5.0]))
    cvs = ds.Canvas(plot_width=7, plot_height=7)
    kwargs = dict(source=df, x=np.arange(3), y=['y0', 'y1', 'y2'], axis=1, line_width=1.0)
    sol_first = np.array([[[2, -1], [2, -1], [1, -1], [1, 1], [1, -1], [-1, -1], [0, -1]], [[2, -1], [2, -1], [1, 2], [1, -1], [1, -1], [0, 1], [0, -1]], [[-1, -1], [1, 2], [1, 2], [2, -1], [-1, -1], [0, 1], [0, 1]], [[0, -1], [1, -1], [1, 2], [2, 2], [0, 2], [0, -1], [1, -1]], [[0, 1], [0, 1], [-1, -1], [2, -1], [0, 2], [0, 2], [-1, -1]], [[1, -1], [0, 1], [0, -1], [0, -1], [0, 2], [2, -1], [2, -1]], [[1, -1], [-1, -1], [0, -1], [0, 0], [0, -1], [2, -1], [2, -1]]], dtype=int)
    sol_last = np.array([[[2, -1], [2, -1], [1, -1], [1, 1], [1, -1], [-1, -1], [0, -1]], [[2, -1], [2, -1], [2, 1], [1, -1], [1, -1], [1, 0], [0, -1]], [[-1, -1], [2, 1], [2, 1], [2, -1], [-1, -1], [1, 0], [1, 0]], [[0, -1], [1, -1], [2, 1], [2, 2], [2, 0], [0, -1], [1, -1]], [[1, 0], [1, 0], [-1, -1], [2, -1], [2, 0], [2, 0], [-1, -1]], [[1, -1], [1, 0], [0, -1], [0, -1], [2, 0], [2, -1], [2, -1]], [[1, -1], [-1, -1], [0, -1], [0, 0], [0, -1], [2, -1], [2, -1]]], dtype=int)
    sol_min = np.array([[[2, -1], [2, -1], [1, -1], [1, 1], [1, -1], [-1, -1], [0, -1]], [[2, -1], [2, -1], [2, 1], [1, -1], [1, -1], [0, 1], [0, -1]], [[-1, -1], [2, 1], [2, 1], [2, -1], [-1, -1], [0, 1], [0, 1]], [[0, -1], [1, -1], [2, 1], [2, 2], [2, 0], [0, -1], [1, -1]], [[1, 0], [0, 1], [-1, -1], [2, -1], [2, 0], [2, 0], [-1, -1]], [[1, -1], [1, 0], [0, -1], [0, -1], [2, 0], [2, -1], [2, -1]], [[1, -1], [-1, -1], [0, -1], [0, 0], [0, -1], [2, -1], [2, -1]]], dtype=int)
    sol_max = np.array([[[2, -1], [2, -1], [1, -1], [1, 1], [1, -1], [-1, -1], [0, -1]], [[2, -1], [2, -1], [1, 2], [1, -1], [1, -1], [1, 0], [0, -1]], [[-1, -1], [1, 2], [1, 2], [2, -1], [-1, -1], [1, 0], [1, 0]], [[0, -1], [1, -1], [1, 2], [2, 2], [0, 2], [0, -1], [1, -1]], [[0, 1], [1, 0], [-1, -1], [2, -1], [0, 2], [0, 2], [-1, -1]], [[1, -1], [0, 1], [0, -1], [0, -1], [0, 2], [2, -1], [2, -1]], [[1, -1], [-1, -1], [0, -1], [0, 0], [0, -1], [2, -1], [2, -1]]], dtype=int)
    sol_index = sol_first
    sol_other = sol_index.choose(np.append(df['other'], nan), mode='wrap')
    agg = cvs.line(agg=ds.where(ds.first('value')), **kwargs)
    assert_eq_ndarray(agg.data, sol_index[:, :, 0])
    agg = cvs.line(agg=ds.where(ds.first('value'), 'other'), **kwargs)
    assert_eq_ndarray(agg.data, sol_other[:, :, 0])
    agg = cvs.line(agg=ds.where(ds.first_n('value', n=2)), **kwargs)
    assert_eq_ndarray(agg.data, sol_index)
    agg = cvs.line(agg=ds.where(ds.first_n('value', n=2), 'other'), **kwargs)
    assert_eq_ndarray(agg.data, sol_other)
    agg = cvs.line(agg=ds.where(ds._min_row_index()), **kwargs)
    assert_eq_ndarray(agg.data, sol_index[:, :, 0])
    agg = cvs.line(agg=ds.where(ds._min_row_index(), 'other'), **kwargs)
    assert_eq_ndarray(agg.data, sol_other[:, :, 0])
    agg = cvs.line(agg=ds.where(ds._min_n_row_index(n=2)), **kwargs)
    assert_eq_ndarray(agg.data, sol_index)
    agg = cvs.line(agg=ds.where(ds._min_n_row_index(n=2), 'other'), **kwargs)
    assert_eq_ndarray(agg.data, sol_other)
    sol_index = sol_last
    sol_other = sol_index.choose(np.append(df['other'], nan), mode='wrap')
    agg = cvs.line(agg=ds.where(ds.last('value')), **kwargs)
    assert_eq_ndarray(agg.data, sol_index[:, :, 0])
    agg = cvs.line(agg=ds.where(ds.last('value'), 'other'), **kwargs)
    assert_eq_ndarray(agg.data, sol_other[:, :, 0])
    agg = cvs.line(agg=ds.where(ds.last_n('value', n=2)), **kwargs)
    assert_eq_ndarray(agg.data, sol_index)
    agg = cvs.line(agg=ds.where(ds.last_n('value', n=2), 'other'), **kwargs)
    assert_eq_ndarray(agg.data, sol_other)
    agg = cvs.line(agg=ds.where(ds._max_row_index()), **kwargs)
    assert_eq_ndarray(agg.data, sol_index[:, :, 0])
    agg = cvs.line(agg=ds.where(ds._max_row_index(), 'other'), **kwargs)
    assert_eq_ndarray(agg.data, sol_other[:, :, 0])
    agg = cvs.line(agg=ds.where(ds._max_n_row_index(n=2)), **kwargs)
    assert_eq_ndarray(agg.data, sol_index)
    agg = cvs.line(agg=ds.where(ds._max_n_row_index(n=2), 'other'), **kwargs)
    assert_eq_ndarray(agg.data, sol_other)
    sol_index = sol_min
    sol_other = sol_index.choose(np.append(df['other'], nan), mode='wrap')
    agg = cvs.line(agg=ds.where(ds.min('value')), **kwargs)
    assert_eq_ndarray(agg.data, sol_index[:, :, 0])
    agg = cvs.line(agg=ds.where(ds.min('value'), 'other'), **kwargs)
    assert_eq_ndarray(agg.data, sol_other[:, :, 0])
    agg = cvs.line(agg=ds.where(ds.min_n('value', n=2)), **kwargs)
    assert_eq_ndarray(agg.data, sol_index)
    agg = cvs.line(agg=ds.where(ds.min_n('value', n=2), 'other'), **kwargs)
    assert_eq_ndarray(agg.data, sol_other)
    sol_index = sol_max
    sol_other = sol_index.choose(np.append(df['other'], nan), mode='wrap')
    agg = cvs.line(agg=ds.where(ds.max('value')), **kwargs)
    assert_eq_ndarray(agg.data, sol_index[:, :, 0])
    agg = cvs.line(agg=ds.where(ds.max('value'), 'other'), **kwargs)
    assert_eq_ndarray(agg.data, sol_other[:, :, 0])
    agg = cvs.line(agg=ds.where(ds.max_n('value', n=2)), **kwargs)
    assert_eq_ndarray(agg.data, sol_index)
    agg = cvs.line(agg=ds.where(ds.max_n('value', n=2), 'other'), **kwargs)
    assert_eq_ndarray(agg.data, sol_other)