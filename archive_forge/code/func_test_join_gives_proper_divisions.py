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
def test_join_gives_proper_divisions():
    df = pd.DataFrame({'a': ['a', 'b', 'c']}, index=[0, 1, 2])
    ddf = dd.from_pandas(df, npartitions=1)
    right_df = pd.DataFrame({'b': [1.0, 2.0, 3.0]}, index=['a', 'b', 'c'])
    expected = df.join(right_df, how='inner', on='a')
    actual = ddf.join(right_df, how='inner', on='a')
    assert actual.divisions == ddf.divisions
    assert_eq(expected, actual)