from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_VERSION, tm
from dask.dataframe.reshape import _get_dummies_dtype_default
from dask.dataframe.utils import assert_eq
def check_pandas_issue_45618_warning(test_func):

    def decorator():
        with warnings.catch_warnings(record=True) as record:
            test_func()
        if PANDAS_VERSION == parse_version('1.4.0'):
            assert all(('In a future version, passing a SparseArray' in str(r.message) for r in record))
        else:
            assert not record
    return decorator