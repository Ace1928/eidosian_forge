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
@pytest.mark.parametrize('df_args,cvs_kwargs', line_manual_range_params)
def test_line_manual_range(DataFrame, df_args, cvs_kwargs):
    if cudf and DataFrame is cudf_DataFrame:
        if isinstance(getattr(df_args[0].get('x', []), 'dtype', ''), RaggedDtype) or (sp and isinstance(getattr(df_args[0].get('geom', []), 'dtype', ''), LineDtype)):
            pytest.skip('cudf DataFrames do not support extension types')
    df = DataFrame(*df_args, geo='geometry' in cvs_kwargs)
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-3.0, 3.0), 7), 7)
    cvs = ds.Canvas(plot_width=7, plot_height=7, x_range=(-3, 3), y_range=(-3, 3))
    agg = cvs.line(df, agg=ds.count(), **cvs_kwargs)
    sol = np.array([[0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, lincoords], dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (-3, 3), close=True)
    assert_eq_ndarray(agg.y_range, (-3, 3), close=True)