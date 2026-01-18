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
def test_categorical_merge_does_not_raise_setting_with_copy_warning():
    df1 = pd.DataFrame(data={'A': ['a', 'b', 'c']}, index=['s', 'v', 'w'])
    df2 = pd.DataFrame(data={'B': ['t', 'd', 'i']}, index=['v', 'w', 'r'])
    ddf1 = dd.from_pandas(df1, npartitions=1)
    df2 = df2.astype({'B': 'category'})
    q = ddf1.join(df2)
    assert_eq(df1.join(df2), q)