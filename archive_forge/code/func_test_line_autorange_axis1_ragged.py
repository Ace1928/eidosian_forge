from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_line_autorange_axis1_ragged():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-4.0, 4.0), 9), 9)
    df = pd.DataFrame({'x': pd.array([[4, 0], [0, -4, 0, 4]], dtype='Ragged[float32]'), 'y': pd.array([[0, -4], [-4, 0, 4, 0]], dtype='Ragged[float32]')})
    cvs = ds.Canvas(plot_width=9, plot_height=9)
    agg = cvs.line(df, 'x', 'y', ds.count(), axis=1)
    sol = np.array([[0, 0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 2], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, lincoords], dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (-4, 4), close=True)
    assert_eq_ndarray(agg.y_range, (-4, 4), close=True)