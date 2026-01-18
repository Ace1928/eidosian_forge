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
def test_groupby_apply_tasks(shuffle_method):
    if shuffle_method == 'disk':
        pytest.skip('Tasks-only shuffle test')
    df = _compat.makeTimeDataFrame()
    df['A'] = df.A // 0.1
    df['B'] = df.B // 0.1
    ddf = dd.from_pandas(df, npartitions=10)
    for ind in [lambda x: 'A', lambda x: x.A]:
        a = df.groupby(ind(df)).apply(len, **INCLUDE_GROUPS)
        with pytest.warns(UserWarning):
            b = ddf.groupby(ind(ddf)).apply(len, **INCLUDE_GROUPS)
        assert_eq(a, b.compute())
        assert not any(('partd' in k[0] for k in b.dask))
        a = df.groupby(ind(df)).B.apply(len, **INCLUDE_GROUPS)
        with pytest.warns(UserWarning):
            b = ddf.groupby(ind(ddf)).B.apply(len, **INCLUDE_GROUPS)
        assert_eq(a, b.compute())
        assert not any(('partd' in k[0] for k in b.dask))