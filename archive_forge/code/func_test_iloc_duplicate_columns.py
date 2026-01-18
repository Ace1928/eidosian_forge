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
@pytest.mark.xfail(DASK_EXPR_ENABLED, reason='duplicated columns')
def test_iloc_duplicate_columns():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    ddf = dd.from_pandas(df, 2)
    df.columns = ['A', 'A', 'C']
    ddf.columns = ['A', 'A', 'C']
    selection = ddf.iloc[:, 2]
    assert any([key.startswith('iloc') for key in selection.dask.layers.keys()])
    select_first = ddf.iloc[:, 1]
    assert_eq(select_first, df.iloc[:, 1])
    select_zeroth = ddf.iloc[:, 0]
    assert_eq(select_zeroth, df.iloc[:, 0])
    select_list_cols = ddf.iloc[:, [0, 2]]
    assert_eq(select_list_cols, df.iloc[:, [0, 2]])
    select_negative = ddf.iloc[:, -1:-3:-1]
    assert_eq(select_negative, df.iloc[:, -1:-3:-1])