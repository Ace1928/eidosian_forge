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
@pytest.mark.parametrize('spec', [{'c': 'median'}, {'b': 'median', 'c': 'max'}])
@pytest.mark.parametrize('keys', ['a', ['a', 'd']])
def test_aggregate_median(spec, keys, shuffle_method):
    pdf = pd.DataFrame({'a': [1, 2, 3, 1, 1, 2, 4, 3, 7] * 10, 'b': [4, 2, 7, 3, 3, 1, 1, 1, 2] * 10, 'c': [0, 1, 2, 3, 4, 5, 6, 7, 8] * 10, 'd': [3, 2, 1, 3, 2, 1, 2, 6, 4] * 10}, columns=['c', 'b', 'a', 'd'])
    ddf = dd.from_pandas(pdf, npartitions=10)
    actual = ddf.groupby(keys).aggregate(spec, shuffle_method=shuffle_method)
    expected = pdf.groupby(keys).aggregate(spec)
    assert_eq(actual, expected)
    if not DASK_EXPR_ENABLED:
        with pytest.raises(ValueError, match='must use shuffl'):
            ddf.groupby(keys).aggregate(spec, shuffle_method=False).compute()
        with pytest.raises(ValueError, match='must use shuffl'):
            ddf.groupby(keys).median(shuffle_method=False).compute()