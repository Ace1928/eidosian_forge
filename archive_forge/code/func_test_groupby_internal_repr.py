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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='architecture different')
def test_groupby_internal_repr():
    pdf = pd.DataFrame({'x': [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], 'y': list('abcbabbcda')})
    ddf = dd.from_pandas(pdf, 3)
    gp = pdf.groupby('y')
    dp = ddf.groupby('y')
    assert isinstance(dp, dd.groupby.DataFrameGroupBy)
    assert isinstance(dp._meta, pd.core.groupby.DataFrameGroupBy)
    assert isinstance(dp.obj, dd.DataFrame)
    assert_eq(dp.obj, gp.obj)
    gp = pdf.groupby('y')['x']
    dp = ddf.groupby('y')['x']
    assert isinstance(dp, dd.groupby.SeriesGroupBy)
    assert isinstance(dp._meta, pd.core.groupby.SeriesGroupBy)
    gp = pdf.groupby('y')[['x']]
    dp = ddf.groupby('y')[['x']]
    assert isinstance(dp, dd.groupby.DataFrameGroupBy)
    assert isinstance(dp._meta, pd.core.groupby.DataFrameGroupBy)
    assert isinstance(dp.obj, dd.DataFrame)
    assert_eq(dp.obj, gp.obj)
    gp = pdf.groupby(pdf.y)['x']
    dp = ddf.groupby(ddf.y)['x']
    assert isinstance(dp, dd.groupby.SeriesGroupBy)
    assert isinstance(dp._meta, pd.core.groupby.SeriesGroupBy)
    gp = pdf.groupby(pdf.y)[['x']]
    dp = ddf.groupby(ddf.y)[['x']]
    assert isinstance(dp, dd.groupby.DataFrameGroupBy)
    assert isinstance(dp._meta, pd.core.groupby.DataFrameGroupBy)
    assert isinstance(dp.obj, dd.DataFrame)
    assert_eq(dp.obj, gp.obj)