from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_auto_range_line():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-10.0, 10.0), 5), 5)
    df = pd.DataFrame({'x': [-10, 0, 10, 0, -10], 'y': [0, 10, 0, -10, 0]})
    cvs = ds.Canvas(plot_width=5, plot_height=5)
    agg = cvs.line(df, 'x', 'y', ds.count())
    sol = np.array([[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [2, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, lincoords], dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (-10, 10), close=True)
    assert_eq_ndarray(agg.y_range, (-10, 10), close=True)