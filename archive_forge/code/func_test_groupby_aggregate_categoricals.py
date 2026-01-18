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
@pytest.mark.parametrize('grouping,agg', [(lambda df: df.drop(columns='category_2').groupby('category_1'), lambda grp: grp.mean()), (lambda df: df.drop(columns='category_2').groupby('category_1'), lambda grp: grp.agg('mean')), (lambda df: df.groupby(['category_1', 'category_2']), lambda grp: grp.mean()), (lambda df: df.groupby(['category_1', 'category_2']), lambda grp: grp.agg('mean'))])
def test_groupby_aggregate_categoricals(grouping, agg):
    pdf = pd.DataFrame({'category_1': pd.Categorical(list('AABBCC')), 'category_2': pd.Categorical(list('ABCABC')), 'value': np.random.uniform(size=6)})
    ddf = dd.from_pandas(pdf, 2)
    with check_observed_deprecation():
        expected = agg(grouping(pdf))
    observed_ctx = pytest.warns(FutureWarning, match='observed') if PANDAS_GE_210 and (not PANDAS_GE_300) else contextlib.nullcontext()
    with observed_ctx:
        result = agg(grouping(ddf))
        assert_eq(result, expected)
    with check_observed_deprecation():
        expected = agg(grouping(pdf)['value'])
    with observed_ctx:
        result = agg(grouping(ddf)['value'])
    assert_eq(result, expected)