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
def test_categorical_last(df):
    solution = np.array([[[4, -1, nan, -3], [12, -13, 14, -11]], [[8, -9, 6, -7], [16, -17, 18, -19]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.last('plusminus')))
        assert_eq_ndarray(agg.data, solution)