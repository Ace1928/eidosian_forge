from __future__ import annotations
import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
import pytest
import dask.array
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
@pytest.mark.parametrize('array_module', array_modules)
def test_curve_quadmesh_autorange(array_module):
    c = ds.Canvas(plot_width=4, plot_height=8)
    coord_array = dask.array if array_module is dask.array else np
    Qx = coord_array.array([[1, 2], [1, 2]])
    Qy = coord_array.array([[1, 1], [4, 2]])
    Z = np.arange(4, dtype='int32').reshape(2, 2)
    da = xr.DataArray(array_module.array(Z), coords={'Qx': (['Y', 'X'], Qx), 'Qy': (['Y', 'X'], Qy)}, dims=['Y', 'X'], name='Z')
    x_coords = np.linspace(0.75, 2.25, 4)
    y_coords = np.linspace(-0.5, 6.5, 8)
    out = xr.DataArray(array_module.array([[nan, nan, nan, nan], [0.0, 0.0, nan, nan], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 3.0, 3.0], [2.0, 2.0, 3.0, nan], [2.0, 2.0, nan, nan], [2.0, 2.0, nan, nan], [2.0, nan, nan, nan]]), coords=dict([('Qx', x_coords), ('Qy', y_coords)]), dims=['Qy', 'Qx'])
    res = c.quadmesh(da, x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (0.5, 2.5), close=True)
    assert_eq_ndarray(res.y_range, (-1, 7), close=True)
    res = c.quadmesh(da.transpose('X', 'Y', transpose_coords=True), x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    assert_eq_ndarray(res.x_range, (0.5, 2.5), close=True)
    assert_eq_ndarray(res.y_range, (-1, 7), close=True)