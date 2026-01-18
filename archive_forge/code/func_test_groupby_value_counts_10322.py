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
def test_groupby_value_counts_10322():
    """Repro case for https://github.com/dask/dask/issues/10322."""
    df = pd.DataFrame({'x': [10] * 5 + [6] * 5 + [3] * 5, 'y': [1] * 3 + [2] * 3 + [4] * 3 + [5] * 3 + [2] * 3})
    counts = df.groupby('x')['y'].value_counts()
    ddf = dd.from_pandas(df, npartitions=3)
    dcounts = ddf.groupby('x')['y'].value_counts()
    assert_eq(counts, dcounts)