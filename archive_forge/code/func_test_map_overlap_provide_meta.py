from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
def test_map_overlap_provide_meta():
    df = pd.DataFrame({'x': [1, 2, 4, 7, 11], 'y': [1.0, 2.0, 3.0, 4.0, 5.0]}).rename_axis('myindex')
    ddf = dd.from_pandas(df, npartitions=2)
    res = ddf.map_overlap(lambda df: df.rolling(2).sum(), 2, 0, meta={'x': 'i8', 'y': 'i8'})
    sol = df.rolling(2).sum()
    assert_eq(res, sol)