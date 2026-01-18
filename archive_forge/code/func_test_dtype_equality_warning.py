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
def test_dtype_equality_warning():
    df1 = pd.DataFrame({'a': np.array([1, 2], dtype=np.dtype(np.int64))})
    df2 = pd.DataFrame({'a': np.array([1, 2], dtype=np.dtype(np.longlong))})
    with warnings.catch_warnings(record=True) as record:
        dd.multi.warn_dtype_mismatch(df1, df2, 'a', 'a')
    assert not record