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
@pytest.mark.parametrize('func', ['var', list])
def test_groupby_select_column_agg(func):
    pdf = pd.DataFrame({'A': [1, 2, 3, 1, 2, 3, 1, 2, 4], 'B': [-0.776, -0.4, -0.873, 0.054, 1.419, -0.948, -0.967, -1.714, -0.666]})
    ddf = dd.from_pandas(pdf, npartitions=4)
    actual = ddf.groupby('A')['B'].agg(func)
    expected = pdf.groupby('A')['B'].agg(func)
    assert_eq(actual, expected)