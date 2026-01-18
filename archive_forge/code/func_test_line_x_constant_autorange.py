from __future__ import annotations
import os
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask.context import config
from numpy import nan
import datashader as ds
from datashader.datatypes import RaggedArray
import datashader.utils as du
import pytest
from datashader.tests.test_pandas import (
@pytest.mark.parametrize('DataFrame', DataFrames)
def test_line_x_constant_autorange(DataFrame):
    x = np.array([-4, 0, 4])
    y = ['y0', 'y1', 'y2']
    ax = 1
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((-4.0, 4.0), 9), 9)
    ddf = DataFrame({'y0': [0, 0, 0], 'y1': [-4, 0, 4], 'y2': [0, 0, 0]})
    cvs = ds.Canvas(plot_width=9, plot_height=9)
    agg = cvs.line(ddf, x, y, ds.count(), axis=ax)
    sol = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [3, 1, 1, 1, 1, 1, 1, 1, 3], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords, lincoords], dims=['y', 'x'])
    assert_eq_xr(agg, out)