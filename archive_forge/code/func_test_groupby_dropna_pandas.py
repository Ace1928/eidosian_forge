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
@pytest.mark.parametrize('dropna', [False, True])
def test_groupby_dropna_pandas(dropna):
    df = pd.DataFrame({'a': [1, 2, 3, 4, None, None, 7, 8], 'e': [4, 5, 6, 3, 2, 1, 0, 0]})
    ddf = dd.from_pandas(df, npartitions=3)
    dask_result = ddf.groupby('a', dropna=dropna).e.sum()
    pd_result = df.groupby('a', dropna=dropna).e.sum()
    assert_eq(dask_result, pd_result)