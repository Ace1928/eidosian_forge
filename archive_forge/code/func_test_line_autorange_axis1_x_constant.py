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
def test_line_autorange_axis1_x_constant(DataFrame):
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-4.0, 4.0), 9), 9)
    xs = np.array([-4, 0, 4])
    df = DataFrame({'y0': [0, 0], 'y1': [-4, 4], 'y2': [0, 0]})
    cvs = ds.Canvas(plot_width=9, plot_height=9)
    agg = cvs.line(df, xs, ['y0', 'y1', 'y2'], ds.count(), axis=1)
    sol = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [2, 0, 0, 0, 0, 0, 0, 0, 2], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, lincoords], dims=['y', 'x'])
    assert_eq_xr(agg, out)