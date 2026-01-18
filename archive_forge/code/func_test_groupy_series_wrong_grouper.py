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
def test_groupy_series_wrong_grouper():
    df = pd.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7] * 10, 'b': [4, 2, 7, 3, 3, 1, 1, 1, 2] * 10, 'c': [0, 1, 2, 3, 4, 5, 6, 7, 8] * 10}, columns=['c', 'b', 'a'])
    df = dd.from_pandas(df, npartitions=3)
    s = df['a']
    s.groupby(s)
    s.groupby([s, s])
    with pytest.raises(KeyError):
        s.groupby('foo')
    with pytest.raises(KeyError):
        s.groupby([s, 'foo'])
    with pytest.raises(ValueError):
        s.groupby(df)
    with pytest.raises(ValueError):
        s.groupby([s, df])