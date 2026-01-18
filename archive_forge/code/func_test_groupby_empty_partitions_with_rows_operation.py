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
@pytest.mark.filterwarnings('ignore:Invalid value encountered:RuntimeWarning')
@pytest.mark.parametrize('operation', ['head', 'tail'])
def test_groupby_empty_partitions_with_rows_operation(operation):
    df = pd.DataFrame(data=[['a1', 'b1'], ['a1', None], ['a1', 'b1'], [None, None], [None, None], [None, None], ['a3', 'b3'], ['a3', 'b3'], ['a5', 'b5']], columns=['A', 'B'])
    caller = operator.methodcaller(operation, 1)
    expected = caller(df.groupby('A')['B'])
    ddf = dd.from_pandas(df, npartitions=3)
    actual = caller(ddf.groupby('A')['B'])
    assert_eq(expected, actual)