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
def test_groupby_multilevel_agg():
    df = pd.DataFrame({'a': [1, 2, 3, 1, 2, 3], 'b': [1, 2, 1, 4, 2, 1], 'c': [1, 3, 2, 1, 1, 2], 'd': [1, 2, 1, 1, 2, 2]})
    ddf = dd.from_pandas(df, 2)
    sol = df.groupby(['a']).mean()
    res = ddf.groupby(['a']).mean()
    assert_eq(res, sol)
    sol = df.groupby(['a', 'c']).mean()
    res = ddf.groupby(['a', 'c']).mean()
    assert_eq(res, sol)
    sol = df.groupby([df['a'], df['c']]).mean()
    res = ddf.groupby([ddf['a'], ddf['c']]).mean()
    assert_eq(res, sol)