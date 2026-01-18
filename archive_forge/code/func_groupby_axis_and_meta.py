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
@contextlib.contextmanager
def groupby_axis_and_meta(axis=0):
    with pytest.warns() as record:
        yield
    expected_len = 2 if PANDAS_GE_210 and (not DASK_EXPR_ENABLED) else 1
    if axis == 1:
        expected_len += 1
    assert expected_len, [x.message for x in record.list]
    assert record[-1].category is UserWarning
    assert '`meta` is not specified' in str(record[-1].message)
    if PANDAS_GE_210 and (not DASK_EXPR_ENABLED):
        assert record[0].category is FutureWarning
        assert 'axis' in str(record[0].message)