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
@pytest.mark.parametrize('keyword', ['split_every', 'split_out'])
def test_groupby_reduction_split(keyword, agg_func, shuffle_method):
    if agg_func in {'first', 'last'} and shuffle_method == 'disk':
        pytest.skip(reason='https://github.com/dask/dask/issues/10034')
    pdf = pd.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7] * 100, 'b': [4, 2, 7, 3, 3, 1, 1, 1, 2] * 100})
    ddf = dd.from_pandas(pdf, npartitions=15)

    def call(g, m, **kwargs):
        return getattr(g, m)(**kwargs)
    if agg_func not in ('nunique', 'cov', 'corr'):
        res = call(ddf.groupby('b', sort=False), agg_func, **{keyword: 2})
        sol = call(pdf.groupby('b'), agg_func)
        assert_eq(res, sol)
        assert call(ddf.groupby('b'), agg_func)._name != res._name
    if agg_func == 'var':
        res = call(ddf.groupby('b', sort=False), 'var', ddof=2, **{keyword: 2})
        sol = call(pdf.groupby('b'), 'var', ddof=2)
        assert_eq(res, sol)
        assert call(ddf.groupby('b'), 'var', ddof=2)._name != res._name
    if agg_func not in ('cov', 'corr'):
        res = call(ddf.groupby('b', sort=False).a, agg_func, **{keyword: 2})
        sol = call(pdf.groupby('b').a, agg_func)
        assert_eq(res, sol)
        assert call(ddf.groupby('b').a, agg_func)._name != res._name
    if agg_func == 'var':
        res = call(ddf.groupby('b', sort=False).a, 'var', ddof=2, **{keyword: 2})
        sol = call(pdf.groupby('b').a, 'var', ddof=2)
        assert_eq(res, sol)
        assert call(ddf.groupby('b').a, 'var', ddof=2)._name != res._name
    if agg_func not in ('cov', 'corr'):
        res = call(ddf.a.groupby(ddf.b, sort=False), agg_func, **{keyword: 2})
        sol = call(pdf.a.groupby(pdf.b), agg_func)
        assert_eq(res, sol, check_names=False)
        assert call(ddf.a.groupby(ddf.b), agg_func)._name != res._name
    if agg_func == 'var':
        res = call(ddf.a.groupby(ddf.b, sort=False), 'var', ddof=2, **{keyword: 2})
        sol = call(pdf.a.groupby(pdf.b), 'var', ddof=2)
        assert_eq(res, sol)
        assert call(ddf.a.groupby(ddf.b), 'var', ddof=2)._name != res._name