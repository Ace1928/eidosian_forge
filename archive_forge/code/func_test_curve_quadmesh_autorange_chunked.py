from __future__ import annotations
import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
import pytest
import dask.array
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
def test_curve_quadmesh_autorange_chunked():
    c = ds.Canvas(plot_width=4, plot_height=8)
    Qx = np.array([[1, 2], [1, 2]])
    Qy = np.array([[1, 1], [4, 2]])
    Z = np.arange(4, dtype='int32').reshape(2, 2)
    da = xr.DataArray(np.array(Z), coords={'Qx': (['Y', 'X'], Qx), 'Qy': (['Y', 'X'], Qy)}, dims=['Y', 'X'], name='Z').chunk({'X': 2, 'Y': 1})
    x_coords = np.linspace(0.75, 2.25, 4)
    y_coords = np.linspace(-0.5, 6.5, 8)
    out = xr.DataArray(np.array([[nan, nan, nan, nan], [0.0, 0.0, nan, nan], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 3.0, 3.0], [2.0, 2.0, 3.0, nan], [2.0, 2.0, nan, nan], [2.0, 2.0, nan, nan], [2.0, nan, nan, nan]]), coords=dict([('Qx', x_coords), ('Qy', y_coords)]), dims=['Qy', 'Qx'])
    res = c.quadmesh(da, x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (0.5, 2.5), close=True)
    assert_eq_ndarray(res.y_range, (-1, 7), close=True)
    res = c.quadmesh(da.transpose('X', 'Y', transpose_coords=True), x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (0.5, 2.5), close=True)
    assert_eq_ndarray(res.y_range, (-1, 7), close=True)