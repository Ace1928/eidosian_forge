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
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('grouper', [lambda df: ['a'], lambda df: ['a', 'b'], lambda df: df['a'], lambda df: [df['a'], df['b']], lambda df: [df['a'] > 2, df['b'] > 1]])
@pytest.mark.parametrize('split_out', [1, 2])
def test_dataframe_aggregations_multilevel(grouper, agg_func, split_out):
    sort = split_out == 1

    def call(g, m, **kwargs):
        return getattr(g, m)(**kwargs)
    pdf = pd.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7] * 10, 'b': [4, 2, 7, 3, 3, 1, 1, 1, 2] * 10, 'd': [0, 1, 2, 3, 4, 5, 6, 7, 8] * 10, 'c': [0, 1, 2, 3, 4, 5, 6, 7, 8] * 10}, columns=['c', 'b', 'a', 'd'])
    ddf = dd.from_pandas(pdf, npartitions=10)
    if agg_func not in ('cov', 'corr'):
        assert_eq(call(pdf.groupby(grouper(pdf), sort=sort)['c'], agg_func), call(ddf.groupby(grouper(ddf), sort=sort)['c'], agg_func, split_out=split_out, split_every=2))
    if agg_func != 'nunique':
        if agg_func in ('cov', 'corr') and split_out > 1:
            pytest.skip('https://github.com/dask/dask/issues/9509')
        assert_eq(call(pdf.groupby(grouper(pdf), sort=sort)[['c', 'd']], agg_func), call(ddf.groupby(grouper(ddf), sort=sort)[['c', 'd']], agg_func, split_out=split_out, split_every=2))
        if agg_func in ('cov', 'corr'):
            df = call(pdf.groupby(grouper(pdf), sort=sort), agg_func).sort_index()
            cols = sorted(list(df.columns))
            df = df[cols]
            dddf = call(ddf.groupby(grouper(ddf), sort=sort), agg_func, split_out=split_out, split_every=2).compute()
            dddf = dddf.sort_index()
            cols = sorted(list(dddf.columns))
            dddf = dddf[cols]
            assert_eq(df, dddf)
        else:
            assert_eq(call(pdf.groupby(grouper(pdf), sort=sort), agg_func), call(ddf.groupby(grouper(ddf), sort=sort), agg_func, split_out=split_out, split_every=2))