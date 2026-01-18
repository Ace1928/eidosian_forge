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
def test_max_n(df):
    solution = np.array([[[4, 0, -1, -3, nan, nan], [14, 12, 10, -11, -13, nan]], [[8, 6, -5, -7, -9, nan], [18, 16, -15, -17, -19, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.max_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.max('plusminus')).data)