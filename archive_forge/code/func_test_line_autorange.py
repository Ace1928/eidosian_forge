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
@pytest.mark.parametrize('df_args,cvs_kwargs', line_autorange_params)
@pytest.mark.parametrize('line_width', [0, 1])
def test_line_autorange(DataFrame, df_args, cvs_kwargs, line_width):
    if cudf and DataFrame is cudf_DataFrame:
        if isinstance(getattr(df_args[0].get('x', []), 'dtype', ''), RaggedDtype) or (sp and isinstance(getattr(df_args[0].get('geom', []), 'dtype', ''), LineDtype)):
            pytest.skip('cudf DataFrames do not support extension types')
        if line_width > 0:
            pytest.skip('cudf DataFrames do not support antialiased lines')
    df = DataFrame(*df_args, geo='geometry' in cvs_kwargs)
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-4.0, 4.0), 9), 9)
    cvs = ds.Canvas(plot_width=9, plot_height=9)
    agg = cvs.line(df, agg=ds.count(), line_width=line_width, **cvs_kwargs)
    if line_width > 0:
        sol = np.array([[np.nan, np.nan, np.nan, 0.646447, 1.292893, 0.646447, np.nan, np.nan, np.nan], [np.nan, np.nan, 0.646447, 0.646447, np.nan, 0.646447, 0.646447, np.nan, np.nan], [np.nan, 0.646447, 0.646447, np.nan, np.nan, np.nan, 0.646447, 0.646447, np.nan], [0.646447, 0.646447, np.nan, np.nan, np.nan, np.nan, np.nan, 0.646447, 0.646447], [0.646447, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.646447], [0.646447, 0.646447, np.nan, np.nan, np.nan, np.nan, np.nan, 0.646447, 0.646447], [np.nan, 0.646447, 0.646447, np.nan, np.nan, np.nan, 0.646447, 0.646447, np.nan], [np.nan, np.nan, 0.646447, 0.646447, np.nan, 0.646447, 0.646447, np.nan, np.nan], [np.nan, np.nan, np.nan, 0.646447, 1.292893, 0.646447, np.nan, np.nan, np.nan]], dtype='f4')
    else:
        sol = np.array([[0, 0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, lincoords], dims=['y', 'x'])
    assert_eq_xr(agg, out, close=line_width > 0)
    assert_eq_ndarray(agg.x_range, (-4, 4), close=True)
    assert_eq_ndarray(agg.y_range, (-4, 4), close=True)