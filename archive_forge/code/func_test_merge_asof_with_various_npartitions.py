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
def test_merge_asof_with_various_npartitions():
    df = pd.DataFrame(dict(ts=[pd.to_datetime('1-1-2020')] * 3, foo=[1, 2, 3]))
    expected = pd.merge_asof(left=df, right=df, on='ts')
    for npartitions in range(1, 5):
        ddf = dd.from_pandas(df, npartitions=npartitions)
        result = dd.merge_asof(left=ddf, right=ddf, on='ts')
        assert_eq(expected, result)