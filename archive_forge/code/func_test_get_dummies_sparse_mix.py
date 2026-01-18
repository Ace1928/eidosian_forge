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
@check_pandas_issue_45618_warning
def test_get_dummies_sparse_mix():
    df = pd.DataFrame({'A': pd.Categorical(['a', 'b', 'a'], categories=['a', 'b', 'c']), 'B': [0, 0, 1]})
    ddf = dd.from_pandas(df, 2)
    exp = pd.get_dummies(df, sparse=True)
    res = dd.get_dummies(ddf, sparse=True)
    with ignore_numpy_bool8_deprecation():
        assert_eq(exp, res)
    dtype = res.compute().A_a.dtype
    assert dtype.fill_value == _get_dummies_dtype_default(0)
    assert dtype.subtype == _get_dummies_dtype_default
    assert isinstance(res.A_a.compute().dtype, pd.SparseDtype)