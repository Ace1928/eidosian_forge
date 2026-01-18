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
def test_categorical_mean_binning(df):
    sol = np.array([[[2, nan, nan, nan], [nan, nan, 12, nan]], [[nan, 7, nan, nan], [nan, nan, nan, 17]]])
    sol = np.append(sol, [[[nan], [nan]], [[nan], [nan]]], axis=2)
    for col in ('f32', 'f64'):
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=dims + [col])
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.mean(col)))
        assert_eq_xr(agg, out)