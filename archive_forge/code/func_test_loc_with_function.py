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
def test_loc_with_function():
    assert_eq(d.loc[lambda df: df['a'] > 3, :], full.loc[lambda df: df['a'] > 3, :])

    def _col_loc_fun(_df):
        return _df.columns.str.contains('b')
    assert_eq(d.loc[:, _col_loc_fun], full.loc[:, _col_loc_fun])