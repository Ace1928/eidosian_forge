from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('df_kwargs, cvs_kwargs, sol_False, sol_True', line_antialias_nan_params)
@pytest.mark.parametrize('self_intersect', [False, True])
def test_line_antialias_nan(df_kwargs, cvs_kwargs, sol_False, sol_True, self_intersect):
    x_range = y_range = (-1, 11)
    cvs = ds.Canvas(plot_width=12, plot_height=12, x_range=x_range, y_range=y_range)
    if 'geometry' in cvs_kwargs:
        df = sp.GeoDataFrame(df_kwargs)
    else:
        df = pd.DataFrame(**df_kwargs)
    agg = cvs.line(df, line_width=2, agg=ds.count(self_intersect=self_intersect), **cvs_kwargs)
    sol = sol_True if self_intersect else sol_False
    assert_eq_ndarray(agg.data, sol, close=True)
    assert_eq_ndarray(agg.x_range, x_range, close=True)
    assert_eq_ndarray(agg.y_range, y_range, close=True)