from __future__ import annotations
import os
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask.context import config
from numpy import nan
import datashader as ds
from datashader.datatypes import RaggedArray
import datashader.utils as du
import pytest
from datashader.tests.test_pandas import (
@pytest.mark.parametrize('on_gpu', [False, True])
def test_dask_categorical_counts(on_gpu):
    if on_gpu and (not test_gpu):
        pytest.skip('gpu tests not enabled')
    df = pd.DataFrame(data=dict(x=[0, 1, 2, 0, 1, 2, 1, 1, 1, 1, 1, 1], y=[0] * 12, cat=['a', 'b', 'c', 'a', 'b', 'c', 'b', 'b', 'b', 'b', 'b', 'c']))
    ddf = dd.from_pandas(df, npartitions=2)
    assert ddf.npartitions == 2
    ddf['cat'] = ddf.cat.astype('category')
    cat_totals = ddf.cat.value_counts().compute()
    assert cat_totals['a'] == 2
    assert cat_totals['b'] == 7
    assert cat_totals['c'] == 3
    canvas = ds.Canvas(3, 1, x_range=(0, 2), y_range=(-1, 1))
    agg = canvas.points(ddf, 'x', 'y', ds.by('cat', ds.count()))
    assert all(agg.cat == ['a', 'b', 'c'])
    sum_cat = agg.sum(dim=['x', 'y'])
    assert all(sum_cat.cat == ['a', 'b', 'c'])
    assert all(sum_cat.values == [2, 7, 3])