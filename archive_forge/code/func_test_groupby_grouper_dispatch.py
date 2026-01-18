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
@pytest.mark.gpu
@pytest.mark.parametrize('key', ['a', 'b'])
def test_groupby_grouper_dispatch(key):
    cudf = pytest.importorskip('cudf')
    pytest.importorskip('dask_cudf')
    pdf = pd.DataFrame({'a': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], 'b': [1, 2, 3, 4, 5, 6, 7, 8], 'c': [1.0, 2.0, 3.5, 4.1, 5.5, 6.6, 7.9, 8.8]})
    gdf = cudf.from_pandas(pdf)
    pd_grouper = grouper_dispatch(pdf)(key=key)
    gd_grouper = grouper_dispatch(gdf)(key=key)
    expect = pdf.groupby(pd_grouper).sum(numeric_only=True)
    got = gdf.groupby(gd_grouper).sum()
    assert_eq(expect, got)