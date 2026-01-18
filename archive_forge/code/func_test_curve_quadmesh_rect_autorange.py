from __future__ import annotations
import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
import pytest
import dask.array
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
@pytest.mark.parametrize('array_module', array_modules)
def test_curve_quadmesh_rect_autorange(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4)
    coord_array = dask.array if array_module is dask.array else np
    Qx = coord_array.array([[1, 2], [1, 2]])
    Qy = coord_array.array([[1, 1], [2, 2]])
    Z = np.arange(4, dtype='int32').reshape(2, 2)
    da = xr.DataArray(array_module.array(Z), coords={'Qx': (['Y', 'X'], Qx), 'Qy': (['Y', 'X'], Qy)}, dims=['Y', 'X'], name='Z')
    y_coords = np.linspace(0.75, 2.25, 4)
    x_coords = np.linspace(0.625, 2.375, 8)
    out = xr.DataArray(array_module.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]], dtype='f8'), coords=[('Qy', y_coords), ('Qx', x_coords)])
    res = c.quadmesh(da, x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    res = c.quadmesh(da.transpose('X', 'Y', transpose_coords=True), x='Qx', y='Qy', agg=ds.sum('Z'))
    assert_eq_xr(res, out)