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
def test_groupby_agg_custom__mode():

    def agg_mode(s):

        def impl(s):
            res, = s.iloc[0]
            for i, in s.iloc[1:]:
                res = res.add(i, fill_value=0)
            return [res]
        return s.apply(impl, **INCLUDE_GROUPS)
    agg_func = dd.Aggregation('custom_mode', lambda s: s.apply(lambda s: [s.value_counts()], **INCLUDE_GROUPS), agg_mode, lambda s: s.map(lambda i: i[0].idxmax()))
    d = pd.DataFrame({'g0': [0, 0, 0, 1, 1] * 3, 'g1': [0, 0, 0, 1, 1] * 3, 'cc': [4, 5, 4, 6, 6] * 3})
    a = dd.from_pandas(d, npartitions=5)
    actual = a['cc'].groupby([a['g0'], a['g1']]).agg(agg_func)
    expected = pd.DataFrame({'g0': [0, 1], 'g1': [0, 1], 'cc': [4, 6]})
    expected = expected['cc'].groupby([expected['g0'], expected['g1']]).agg('sum')
    assert_eq(actual, expected)