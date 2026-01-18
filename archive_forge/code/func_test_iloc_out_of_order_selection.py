from __future__ import annotations
import contextlib
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import tokenize
from dask.dataframe._compat import PANDAS_GE_210, PANDAS_GE_220, IndexingError, tm
from dask.dataframe.indexing import _coerce_loc_index
from dask.dataframe.utils import assert_eq, make_meta, pyarrow_strings_enabled
def test_iloc_out_of_order_selection():
    df = pd.DataFrame({'A': [1] * 100, 'B': [2] * 100, 'C': [3] * 100, 'D': [4] * 100})
    ddf = dd.from_pandas(df, 2)
    ddf = ddf[['C', 'A', 'B']]
    a = ddf.iloc[:, 0]
    b = ddf.iloc[:, 1]
    c = ddf.iloc[:, 2]
    assert a.name == 'C'
    assert b.name == 'A'
    assert c.name == 'B'
    a1, b1, c1 = dask.compute(a, b, c)
    assert a1.name == 'C'
    assert b1.name == 'A'
    assert c1.name == 'B'