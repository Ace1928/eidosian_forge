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
@pytest.mark.parametrize('indexer', [0, [0], [0, 1], [1, 0], [False, True, True]])
def test_iloc(indexer):
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    ddf = dd.from_pandas(df, 2)
    result = ddf.iloc[:, indexer]
    expected = df.iloc[:, indexer]
    assert_eq(result, expected)