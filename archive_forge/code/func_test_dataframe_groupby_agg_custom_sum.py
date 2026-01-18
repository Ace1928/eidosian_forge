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
@pytest.mark.parametrize('pandas_spec, dask_spec, check_dtype', [({'b': 'mean'}, {'b': custom_mean}, False), ({'b': 'sum'}, {'b': custom_sum}, True), (['mean', 'sum'], [custom_mean, custom_sum], False), ({'b': ['mean', 'sum']}, {'b': [custom_mean, custom_sum]}, False)])
def test_dataframe_groupby_agg_custom_sum(pandas_spec, dask_spec, check_dtype):
    df = pd.DataFrame({'g': [0, 0, 1] * 3, 'b': [1, 2, 3] * 3})
    ddf = dd.from_pandas(df, npartitions=2)
    expected = df.groupby('g').aggregate(pandas_spec)
    result = ddf.groupby('g').aggregate(dask_spec)
    assert_eq(result, expected, check_dtype=check_dtype)