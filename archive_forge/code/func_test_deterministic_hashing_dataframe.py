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
def test_deterministic_hashing_dataframe():
    obj = pd.DataFrame([[0, 1, 2, 3], [4, 5, 6, 7]], columns=['a', 'b', 'c', 'c'])
    dask_df = dd.from_pandas(obj, npartitions=1)
    ddf1 = dask_df.loc[0:1, ['a', 'c']]
    ddf2 = dask_df.loc[0:1, ['a', 'c']]
    assert tokenize(ddf1) == tokenize(ddf2)
    ddf1 = dask_df.loc[0:1, 'c']
    ddf2 = dask_df.loc[0:1, 'c']
    assert tokenize(ddf1) == tokenize(ddf2)
    ddf1 = dask_df.iloc[:, [0, 1]]
    ddf2 = dask_df.iloc[:, [0, 1]]
    assert tokenize(ddf1) == tokenize(ddf2)
    ddf2 = dask_df.iloc[:, [0, 2]]
    assert tokenize(ddf1) != tokenize(ddf2)