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
@pytest.mark.parametrize('scheduler', ['sync', 'threads'])
def test_groupby_on_index(scheduler):
    pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [4, 5, 6, 3, 2, 1, 0, 0, 0]}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
    ddf = dd.from_pandas(pdf, npartitions=3)
    ddf2 = ddf.set_index('a')
    pdf2 = pdf.set_index('a')
    assert_eq(ddf.groupby('a').b.mean(), ddf2.groupby(ddf2.index).b.mean())
    agg = ddf2.groupby('a').agg({'b': 'mean'})
    assert_eq(ddf.groupby('a').b.mean(), agg.b)
    if not DASK_EXPR_ENABLED:
        assert hlg_layer(agg.dask, 'getitem')

    def func(df):
        return df.assign(b=df.b - df.b.mean())

    def func2(df):
        return df[['b']] - df[['b']].mean()

    def func3(df):
        return df.mean()
    with dask.config.set(scheduler=scheduler):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            assert_eq(ddf.groupby('a').apply(func, **INCLUDE_GROUPS), pdf.groupby('a').apply(func, **INCLUDE_GROUPS))
            assert_eq(pdf2.groupby(pdf2.index).apply(func2, **INCLUDE_GROUPS), ddf2.groupby(ddf2.index).apply(func2, **INCLUDE_GROUPS))
            assert_eq(ddf2.b.groupby('a').apply(func3, **INCLUDE_GROUPS), pdf2.b.groupby('a').apply(func3, **INCLUDE_GROUPS))
            assert_eq(ddf2.b.groupby(ddf2.index).apply(func3, **INCLUDE_GROUPS), pdf2.b.groupby(pdf2.index).apply(func3, **INCLUDE_GROUPS))