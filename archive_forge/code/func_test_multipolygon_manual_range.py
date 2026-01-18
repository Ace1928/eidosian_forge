import pytest
import pandas as pd
import numpy as np
import xarray as xr
import datashader as ds
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
import dask.dataframe as dd
@pytest.mark.skipif(not spatialpandas, reason='spatialpandas not installed')
@pytest.mark.parametrize('DataFrame', DataFrames)
def test_multipolygon_manual_range(DataFrame):
    df = DataFrame({'polygons': pd.Series([[[[0, 0, 2, 0, 2, 2, 1, 3, 0, 0], [1, 0.25, 1, 2, 1.75, 0.25, 0.25, 0.25]], [[2.5, 1, 4, 1, 4, 2, 2.5, 2, 2.5, 1]]]], dtype='MultiPolygon[float64]'), 'v': [1]})
    cvs = ds.Canvas(plot_width=16, plot_height=16)
    agg = cvs.polygons(df, geometry='polygons', agg=ds.count())
    axis = ds.core.LinearAxis()
    lincoords_x = axis.compute_index(axis.compute_scale_and_translate((0.0, 4.0), 16), 16)
    lincoords_y = axis.compute_index(axis.compute_scale_and_translate((0.0, 3.0), 16), 16)
    sol = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 4), close=True)
    assert_eq_ndarray(agg.y_range, (0, 3), close=True)