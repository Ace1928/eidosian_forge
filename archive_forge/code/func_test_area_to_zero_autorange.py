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
@pytest.mark.parametrize('df_kwargs,cvs_kwargs', [(dict(data={'x0': [-4, 0], 'x1': [-2, 2], 'x2': [0, 4], 'y0': [0, 0], 'y1': [-4, -4], 'y2': [0, 0]}, dtype='float32'), dict(x=['x0', 'x1', 'x2'], y=['y0', 'y1', 'y2'], axis=1)), (dict(data={'x0': [-4, 0], 'x1': [-2, 2], 'x2': [0, 4]}, dtype='float32'), dict(x=['x0', 'x1', 'x2'], y=np.array([0, -4, 0], dtype='float32'), axis=1)), (dict(data={'x': [-4, -2, 0, 0, 2, 4], 'y': [0, -4, 0, 0, -4, 0]}), dict(x='x', y='y', axis=0)), (dict(data={'x0': [-4, -2, 0], 'x1': [0, 2, 4], 'y0': [0, -4, 0], 'y1': [0, -4, 0]}, dtype='float32'), dict(x=['x0', 'x1'], y=['y0', 'y1'], axis=0)), (dict(data={'x0': [-4, -2, 0], 'x1': [0, 2, 4], 'y0': [0, -4, 0]}, dtype='float32'), dict(x=['x0', 'x1'], y='y0', axis=0)), (dict(data={'x': pd.array([[-4, -2, 0], [0, 2, 4]], dtype='Ragged[float32]'), 'y': pd.array([[0, -4, 0], [0, -4, 0]], dtype='Ragged[float32]')}), dict(x='x', y='y', axis=1))])
def test_area_to_zero_autorange(DataFrame, df_kwargs, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if isinstance(getattr(df_kwargs['data'].get('x', []), 'dtype', ''), RaggedDtype):
            pytest.skip('cudf DataFrames do not support extension types')
    df = DataFrame(**df_kwargs)
    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(axis.compute_scale_and_translate((-4.0, 0.0), 7), 7)
    lincoords_x = axis.compute_index(axis.compute_scale_and_translate((-4.0, 4.0), 13), 13)
    cvs = ds.Canvas(plot_width=13, plot_height=7)
    agg = cvs.area(df, agg=ds.count(), **cvs_kwargs)
    sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (-4, 4), close=True)
    assert_eq_ndarray(agg.y_range, (-4, 0), close=True)