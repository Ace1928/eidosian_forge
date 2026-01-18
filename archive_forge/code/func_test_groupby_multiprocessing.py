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
def test_groupby_multiprocessing():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['1', '1', 'a', 'a', 'a']})
    ddf = dd.from_pandas(df, npartitions=3)
    expected = df.groupby('B').apply(lambda x: x, **INCLUDE_GROUPS)
    meta = to_pyarrow_string(expected) if pyarrow_strings_enabled() else expected
    with dask.config.set(scheduler='processes'):
        assert_eq(ddf.groupby('B').apply(lambda x: x, meta=meta, **INCLUDE_GROUPS), expected)