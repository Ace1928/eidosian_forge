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
@pytest.mark.filterwarnings('ignore:`meta` is not specified')
def test_groupby_shift_within_partition_sorting():
    for _ in range(10):
        df = pd.DataFrame({'a': range(60), 'b': [2, 4, 3, 1] * 15, 'c': [None, 10, 20, None, 30, 40] * 10})
        df = df.set_index('a').sort_index()
        ddf = dd.from_pandas(df, npartitions=6)
        assert_eq(df.groupby('b')['c'].shift(1), ddf.groupby('b')['c'].shift(1), scheduler='threads')