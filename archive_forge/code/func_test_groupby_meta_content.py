from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="don't store nonempty meta in dask-expr")
@pytest.mark.parametrize('grouper', [lambda df: df['a'], lambda df: df['a'] > 2, lambda df: [df['a'], df['b']], lambda df: [df['a'] > 2], pytest.param(lambda df: [df['a'] > 2, df['b'] > 1], marks=pytest.mark.xfail(not PANDAS_GE_150, reason='index dtype does not coincide: boolean != empty'))])
@pytest.mark.parametrize('group_and_slice', [lambda df, grouper: df.groupby(grouper(df)), lambda df, grouper: df['c'].groupby(grouper(df)), lambda df, grouper: df.groupby(grouper(df))['c']])
def test_groupby_meta_content(group_and_slice, grouper):
    pdf = pd.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7] * 10, 'b': [4, 2, 7, 3, 3, 1, 1, 1, 2] * 10, 'c': [0, 1, 2, 3, 4, 5, 6, 7, 8] * 10}, columns=['c', 'b', 'a'])
    ddf = dd.from_pandas(pdf, npartitions=10)
    expected = group_and_slice(pdf, grouper).first().head(0)
    meta = group_and_slice(ddf, grouper)._meta.first()
    meta_nonempty = group_and_slice(ddf, grouper)._meta_nonempty.first().head(0)
    assert_eq(expected, meta)
    assert_eq(expected, meta_nonempty)