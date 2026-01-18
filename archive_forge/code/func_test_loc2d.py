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
def test_loc2d():
    assert_eq(d.loc[5, 'a'], full.loc[5:5, 'a'])
    assert_eq(d.loc[5, ['a']], full.loc[5:5, ['a']])
    assert_eq(d.loc[3:8, 'a'], full.loc[3:8, 'a'])
    assert_eq(d.loc[:8, 'a'], full.loc[:8, 'a'])
    assert_eq(d.loc[3:, 'a'], full.loc[3:, 'a'])
    assert_eq(d.loc[[8], 'a'], full.loc[[8], 'a'])
    assert_eq(d.loc[3:8, ['a']], full.loc[3:8, ['a']])
    assert_eq(d.loc[:8, ['a']], full.loc[:8, ['a']])
    assert_eq(d.loc[3:, ['a']], full.loc[3:, ['a']])
    with pytest.raises(IndexingError):
        d.loc[3, 3, 3]
    with pytest.raises(IndexingError):
        d.a.loc[3, 3]
    with pytest.raises(IndexingError):
        d.a.loc[3:, 3]
    with pytest.raises(IndexingError):
        d.a.loc[d.a % 2 == 0, 3]