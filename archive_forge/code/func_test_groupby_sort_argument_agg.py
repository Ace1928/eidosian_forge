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
@pytest.mark.parametrize('agg', [M.sum, M.prod, M.max, M.min])
@pytest.mark.parametrize('sort', [True, False])
def test_groupby_sort_argument_agg(agg, sort):
    df = pd.DataFrame({'x': [4, 2, 1, 2, 3, 1], 'y': [1, 2, 3, 4, 5, 6]})
    ddf = dd.from_pandas(df, npartitions=3)
    result = agg(ddf.groupby('x', sort=sort))
    result_pd = agg(df.groupby('x', sort=sort))
    assert_eq(result, result_pd)
    if sort:
        assert_eq(result.index, result_pd.index)