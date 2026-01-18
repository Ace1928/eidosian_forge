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
@pytest.mark.skip_with_pyarrow_strings
@pytest.mark.parametrize('int_dtype', ['uint8', 'int32', 'int64'])
def test_groupby_unique(int_dtype):
    rng = np.random.RandomState(42)
    df = pd.DataFrame({'foo': rng.randint(3, size=100), 'bar': rng.randint(10, size=100)}, dtype=int_dtype)
    ddf = dd.from_pandas(df, npartitions=10)
    pd_gb = df.groupby('foo')['bar'].unique()
    dd_gb = ddf.groupby('foo')['bar'].unique()
    assert_eq(dd_gb.explode(), pd_gb.explode())