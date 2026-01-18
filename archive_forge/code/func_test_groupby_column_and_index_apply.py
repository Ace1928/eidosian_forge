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
@pytest.mark.parametrize('group_args', [['idx', 'a'], ['a', 'idx'], ['idx'], 'idx'])
@pytest.mark.parametrize('apply_func', [np.min, np.mean, lambda s, axis=None: np.max(s.values) - np.mean(s.values)])
def test_groupby_column_and_index_apply(group_args, apply_func):
    df = pd.DataFrame({'idx': [1, 1, 1, 2, 2, 2], 'a': [1, 2, 1, 2, 1, 2], 'b': np.arange(6)}).set_index('idx')
    ddf = dd.from_pandas(df, npartitions=df.index.nunique())
    ddf_no_divs = dd.from_pandas(df, npartitions=df.index.nunique(), sort=False).clear_divisions()
    expected = df.groupby(group_args).apply(apply_func, axis=0, **INCLUDE_GROUPS)
    result = ddf.groupby(group_args).apply(apply_func, axis=0, meta=expected, **INCLUDE_GROUPS)
    assert_eq(expected, result, check_divisions=False)
    assert ddf.divisions == result.divisions
    assert len(result.dask) == len(ddf.dask) + ddf.npartitions
    expected = df.groupby(group_args).apply(apply_func, axis=0, **INCLUDE_GROUPS)
    result = ddf_no_divs.groupby(group_args).apply(apply_func, axis=0, meta=expected, **INCLUDE_GROUPS)
    assert_eq(expected, result, check_divisions=False)
    assert ddf_no_divs.divisions == result.divisions
    assert len(result.dask) > len(ddf_no_divs.dask) + ddf_no_divs.npartitions