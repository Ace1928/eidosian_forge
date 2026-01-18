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
@pytest.mark.xfail(DASK_EXPR_ENABLED, reason='axis not supported')
@pytest.mark.filterwarnings('ignore:`meta` is not specified')
@pytest.mark.parametrize('npartitions', [1, 2, 5])
@pytest.mark.parametrize('period', [1, -1, 10])
@pytest.mark.parametrize('axis', [0, 1])
def test_groupby_shift_basic_input(npartitions, period, axis):
    pdf = pd.DataFrame({'a': [0, 0, 1, 1, 2, 2, 3, 3, 3], 'b': [4, 5, 6, 3, 2, 1, 0, 0, 0], 'c': [0, 0, 0, 0, 0, 1, 1, 1, 1]})
    ddf = dd.from_pandas(pdf, npartitions=npartitions)
    with groupby_axis_deprecated(dask_op=False):
        expected = pdf.groupby(['a', 'c']).shift(period, axis=axis)
    with groupby_axis_and_meta(axis):
        result = ddf.groupby(['a', 'c']).shift(period, axis=axis)
    assert_eq(expected, result)
    if DASK_EXPR_ENABLED:
        ctx = contextlib.nullcontext()
    else:
        ctx = pytest.warns(FutureWarning, match='`axis` parameter is deprecated')
    with ctx:
        ddf.groupby(['a', 'c']).shift(period, axis=axis)
    with groupby_axis_deprecated(dask_op=False):
        expected = pdf.groupby(['a']).shift(period, axis=axis)
    with groupby_axis_and_meta(axis):
        result = ddf.groupby(['a']).shift(period, axis=axis)
    assert_eq(expected, result)
    with groupby_axis_deprecated(dask_op=False):
        expected = pdf.groupby(pdf.c).shift(period, axis=axis)
    with groupby_axis_and_meta(axis):
        result = ddf.groupby(ddf.c).shift(period, axis=axis)
    assert_eq(expected, result)
    with ctx:
        ddf.groupby(ddf.c).shift(period, axis=axis)