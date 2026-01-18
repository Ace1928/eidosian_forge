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
def test_groupby_unaligned_index():
    df = pd.DataFrame({'a': np.random.randint(0, 10, 50), 'b': np.random.randn(50), 'c': np.random.randn(50)})
    ddf = dd.from_pandas(df, npartitions=5)
    filtered = df[df.b < 0.5]
    dfiltered = ddf[ddf.b < 0.5]
    ddf_group = dfiltered.groupby(ddf.a)
    ds_group = dfiltered.b.groupby(ddf.a)
    bad = [ddf_group.mean(), ddf_group.var(), ddf_group.b.nunique(), ddf_group.get_group(0), ds_group.mean(), ds_group.var(), ds_group.nunique(), ds_group.get_group(0)]
    for obj in bad:
        with pytest.raises(ValueError):
            obj.compute()

    def add1(x):
        return x + 1
    df_group = filtered.groupby(df.a)
    expected = df_group.apply(add1, **INCLUDE_GROUPS)
    assert_eq(ddf_group.apply(add1, meta=expected, **INCLUDE_GROUPS), expected)
    expected = df_group.b.apply(add1, **INCLUDE_GROUPS)
    assert_eq(ddf_group.b.apply(add1, meta=expected, **INCLUDE_GROUPS), expected)