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
@pytest.mark.parametrize('by', ['a', 'b', 'c', ['a', 'b'], ['a', 'c']])
@pytest.mark.parametrize('agg', ['count', pytest.param('mean', marks=pytest.mark.xfail(PANDAS_GE_200, reason='numeric_only=False not implemented', strict=False)), pytest.param('std', marks=pytest.mark.xfail(PANDAS_GE_200, reason='numeric_only=False not implemented', strict=False))])
@pytest.mark.parametrize('sort', [True, False])
def test_groupby_sort_argument(by, agg, sort):
    df = pd.DataFrame({'a': [1, 2, 3, 4, None, None, 7, 8], 'b': [1, 0] * 4, 'c': ['a', 'b', None, None, 'e', 'f', 'g', 'h'], 'e': [4, 5, 6, 3, 2, 1, 0, 0]})
    ddf = dd.from_pandas(df, npartitions=3)
    gb = ddf.groupby(by, sort=sort)
    gb_pd = df.groupby(by, sort=sort)
    result_1 = getattr(gb, agg)
    result_1_pd = getattr(gb_pd, agg)
    result_2 = getattr(gb.e, agg)
    result_2_pd = getattr(gb_pd.e, agg)
    result_3 = gb.agg({'e': agg})
    result_3_pd = gb_pd.agg({'e': agg})
    if agg == 'mean':
        with record_numeric_only_warnings() as rec_pd:
            expected = result_1_pd().astype('float')
        with record_numeric_only_warnings() as rec_dd:
            result = result_1()
        assert len(rec_pd) == len(rec_dd)
        assert_eq(result, expected)
        with record_numeric_only_warnings() as rec_pd:
            expected = result_2_pd().astype('float')
        with record_numeric_only_warnings() as rec_dd:
            result = result_2()
        assert len(rec_pd) == len(rec_dd)
        assert_eq(result, expected)
        with record_numeric_only_warnings() as rec_pd:
            expected = result_3_pd.astype('float')
        with record_numeric_only_warnings() as rec_dd:
            result = result_3
        assert len(rec_pd) == len(rec_dd)
        assert_eq(result, expected)
    else:
        with record_numeric_only_warnings() as rec_pd:
            expected = result_1_pd()
        with record_numeric_only_warnings() as rec_dd:
            result = result_1()
        assert len(rec_pd) == len(rec_dd)
        assert_eq(result, expected)
        with record_numeric_only_warnings() as rec_pd:
            expected = result_2_pd()
        with record_numeric_only_warnings() as rec_dd:
            result = result_2()
        assert len(rec_pd) == len(rec_dd)
        assert_eq(result, expected)
        with record_numeric_only_warnings() as rec_pd:
            expected = result_3_pd
        with record_numeric_only_warnings() as rec_dd:
            result = result_3
        assert len(rec_pd) == len(rec_dd)
        assert_eq(result, expected)