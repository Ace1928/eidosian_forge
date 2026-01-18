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
def test_categorical_mean(df):
    sol = np.array([[[2, nan, nan, nan], [nan, nan, 12, nan]], [[nan, 7, nan, nan], [nan, nan, nan, 17]]])
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat'])
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.mean('f32')))
    assert_eq_xr(agg, out)
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.mean('f64')))
    assert_eq_xr(agg, out)
    out = xr.DataArray(sol, coords=coords + [range(4)], dims=dims + ['cat_int'])
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.mean('i32')))
    assert_eq_xr(agg, out)
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.mean('i64')))
    assert_eq_xr(agg, out)