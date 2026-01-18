from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.skipif(not sp, reason='spatialpandas not installed')
@pytest.mark.parametrize('geom_data,geom_type', [([0, 0, 1, 1, 2, 0, 0, 0], 'line'), ([[0, 0, 1, 1, 2, 0, 0, 0]], 'multiline'), ([0, 0, 1, 1, 2, 0, 0, 0], 'ring'), ([[0, 0, 1, 1, 2, 0, 0, 0]], 'polygon'), ([[[0, 0, 1, 1, 2, 0, 0, 0]]], 'multipolygon')])
def test_closed_ring_line(geom_data, geom_type):
    gdf = sp.GeoDataFrame({'geometry': sp.GeoSeries([geom_data], dtype=geom_type)})
    cvs = ds.Canvas(plot_width=4, plot_height=4)
    agg = cvs.line(gdf, geometry='geometry', agg=ds.count())
    coords_x = axis.compute_index(axis.compute_scale_and_translate((0.0, 2), 4), 4)
    coords_y = axis.compute_index(axis.compute_scale_and_translate((0.0, 1), 4), 4)
    sol = np.array([[1, 1, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0]])
    out = xr.DataArray(sol, coords=[coords_y, coords_x], dims=['y', 'x'])
    if geom_type.endswith('line'):
        out[0, 0] = 2
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 2), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)