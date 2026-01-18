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
def test_categorical_where_last_n(df):
    sol_rowindex = xr.DataArray([[[[4, 0, -1], [1, -1, -1], [-1, -1, -1], [3, -1, -1]], [[12, -1, -1], [13, -1, -1], [14, 10, -1], [11, -1, -1]]], [[[8, -1, -1], [9, 5, -1], [6, -1, -1], [7, -1, -1]], [[16, -1, -1], [17, -1, -1], [18, -1, -1], [19, 15, -1]]]], coords=coords + [['a', 'b', 'c', 'd'], [0, 1, 2]], dims=dims + ['cat2', 'n'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)
    for n in range(1, 4):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last_n('plusminus', n=n))))
        out = sol_rowindex[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data, c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last('plusminus')))).data)
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last_n('plusminus', n=n), 'reverse')))
        out = sol_reverse[:, :, :, :n]
        assert_eq_xr(agg, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, :, 0].data, c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.last('plusminus'), 'reverse'))).data)