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
def test_line_agg_sum_axis1_none_constant(DataFrame):
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-3.0, 3.0), 7), 7)
    df = DataFrame({'x0': [4, -4], 'x1': [0, 0], 'x2': [-4, 4], 'y0': [0, 0], 'y1': [-4, 4], 'y2': [0, 0], 'v': [7, 9]})
    cvs = ds.Canvas(plot_width=7, plot_height=7, x_range=(-3, 3), y_range=(-3, 3))
    agg = cvs.line(df, ['x0', 'x1', 'x2'], ['y0', 'y1', 'y2'], ds.sum('v'), axis=1)
    nan = np.nan
    sol = np.array([[nan, nan, 7, nan, 7, nan, nan], [nan, 7, nan, nan, nan, 7, nan], [7, nan, nan, nan, nan, nan, 7], [nan, nan, nan, nan, nan, nan, nan], [9, nan, nan, nan, nan, nan, 9], [nan, 9, nan, nan, nan, 9, nan], [nan, nan, 9, nan, 9, nan, nan]], dtype='float32')
    out = xr.DataArray(sol, coords=[lincoords, lincoords], dims=['y', 'x'])
    assert_eq_xr(agg, out)