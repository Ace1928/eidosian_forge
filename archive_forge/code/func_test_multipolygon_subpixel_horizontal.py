import pytest
import pandas as pd
import numpy as np
import xarray as xr
import datashader as ds
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
import dask.dataframe as dd
@pytest.mark.skipif(not spatialpandas, reason='spatialpandas not installed')
@pytest.mark.parametrize('DataFrame', DataFrames)
@pytest.mark.parametrize('scale', [4, 100])
def test_multipolygon_subpixel_horizontal(DataFrame, scale):
    df = GeoDataFrame({'geometry': MultiPolygonArray([[[[0, 0, 1, 0, 1, 1, 0, 1, 0, 0]], [[0, 2, 1, 2, 1, 3, 0, 3, 0, 2]]]])})
    cvs = ds.Canvas(plot_height=8, plot_width=8, x_range=(-2 * scale, 2 * scale), y_range=(0, 4))
    agg = cvs.polygons(df, 'geometry', agg=ds.count())
    sol = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
    axis = ds.core.LinearAxis()
    lincoords_x = axis.compute_index(axis.compute_scale_and_translate((-2 * scale, 2 * scale), 8), 8)
    lincoords_y = axis.compute_index(axis.compute_scale_and_translate((0, 4), 8), 8)
    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])
    assert_eq_xr(agg, out)