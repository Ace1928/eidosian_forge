from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.api.types import is_object_dtype
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.core import _Frame
from dask.dataframe.methods import concat
from dask.dataframe.multi import (
from dask.dataframe.utils import (
from dask.utils_test import hlg_layer, hlg_layer_topological
def test_categorical_merge_with_columns_missing_from_left():
    df1 = pd.DataFrame({'A': [0, 1], 'B': pd.Categorical(['a', 'b'])})
    df2 = pd.DataFrame({'C': pd.Categorical(['a', 'b'])})
    expected = pd.merge(df2, df1, left_index=True, right_on='A')
    ddf1 = dd.from_pandas(df1, npartitions=2)
    ddf2 = dd.from_pandas(df2, npartitions=2)
    actual = dd.merge(ddf2, ddf1, left_index=True, right_on='A').compute()
    assert actual.C.dtype == 'category'
    assert actual.B.dtype == 'category'
    assert actual.A.dtype == 'int64'
    assert actual.index.dtype == 'int64'
    assert assert_eq(expected, actual)