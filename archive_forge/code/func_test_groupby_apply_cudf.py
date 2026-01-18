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
@pytest.mark.gpu
@pytest.mark.parametrize('group_keys', [pytest.param(True, marks=pytest.mark.skipif(not PANDAS_GE_150, reason='cudf and pandas behave differently')), False])
def test_groupby_apply_cudf(group_keys):
    pytest.importorskip('dask_cudf')
    cudf = pytest.importorskip('cudf')
    df = pd.DataFrame({'a': [1, 2, 3, 1, 2, 3], 'b': [4, 5, 6, 7, 8, 9]})
    ddf = dd.from_pandas(df, npartitions=2)
    dcdf = ddf.to_backend('cudf')
    func = lambda x: x
    res_pd = df.groupby('a', group_keys=group_keys).apply(func, **INCLUDE_GROUPS)
    res_dd = ddf.groupby('a', group_keys=group_keys).apply(func, meta=res_pd, **INCLUDE_GROUPS)
    res_dc = dcdf.groupby('a', group_keys=group_keys).apply(func, meta=cudf.from_pandas(res_pd), **INCLUDE_GROUPS)
    assert_eq(res_pd, res_dd)
    assert_eq(res_dd, res_dc)