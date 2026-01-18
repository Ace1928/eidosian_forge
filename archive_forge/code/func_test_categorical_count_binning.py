from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('df', dfs)
def test_categorical_count_binning(df):
    sol = np.array([[[5, 0, 0, 0], [0, 0, 5, 0]], [[0, 5, 0, 0], [0, 0, 0, 5]]])
    sol = np.append(sol, [[[0], [0]], [[0], [0]]], axis=2)
    for col in ('i32', 'i64'):
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=dims + [col])
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.count()))
        assert_eq_xr(agg, out)
    sol[0, 0, 0] = 4
    sol[0, 0, 4] = 1
    for col in ('f32', 'f64'):
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=dims + [col])
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.count()))
        assert_eq_xr(agg, out)