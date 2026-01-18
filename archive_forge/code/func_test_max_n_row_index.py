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
def test_max_n_row_index(df):
    solution = np.array([[[4, 3, 2, 1, 0, -1], [14, 13, 12, 11, 10, -1]], [[9, 8, 7, 6, 5, -1], [19, 18, 17, 16, 15, -1]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds._max_n_row_index(n=n))
        assert agg.dtype == np.int64
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds._max_row_index()).data)