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
@pytest.mark.parametrize('dropna', [False, True, None])
@pytest.mark.parametrize('by', ['a', 'c', 'd', ['a', 'b'], ['a', 'c'], ['a', 'd']])
@pytest.mark.parametrize('group_keys', [True, False, None])
def test_groupby_dropna_cudf(dropna, by, group_keys):
    cudf = pytest.importorskip('cudf')
    dask_cudf = pytest.importorskip('dask_cudf')
    df = cudf.DataFrame({'a': [1, 2, 3, 4, None, None, 7, 8], 'b': [1, 0] * 4, 'c': ['a', 'b', None, None, 'e', 'f', 'g', 'h'], 'e': [4, 5, 6, 3, 2, 1, 0, 0]})
    df['d'] = df['c'].astype('category')
    ddf = dask_cudf.from_cudf(df, npartitions=3)
    if dropna is None:
        dask_result = ddf.groupby(by, group_keys=group_keys).e.sum()
        cudf_result = df.groupby(by, group_keys=group_keys).e.sum()
    else:
        dask_result = ddf.groupby(by, dropna=dropna, group_keys=group_keys).e.sum()
        cudf_result = df.groupby(by, dropna=dropna, group_keys=group_keys).e.sum()
    if by in ['c', 'd']:
        dask_result = dask_result.compute()
        dask_result.index.name = cudf_result.index.name
    assert_eq(dask_result, cudf_result)