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
@pytest.mark.parametrize('by', ['key1', ['key1', 'key2']])
@pytest.mark.parametrize('slice_key', [3, 'value', ['value'], ('value',), pd.Index(['value']), pd.Series(['value'])])
def test_groupby_slice_getitem(by, slice_key):
    pdf = pd.DataFrame({'key1': ['a', 'b', 'a'], 'key2': ['c', 'c', 'c'], 'value': [1, 2, 3], 3: [1, 2, 3]})
    ddf = dd.from_pandas(pdf, npartitions=3)
    expect = pdf.groupby(by)[slice_key].count()
    got = ddf.groupby(by)[slice_key].count()
    if not DASK_EXPR_ENABLED:
        assert hlg_layer(got.dask, 'getitem')
    assert_eq(expect, got)