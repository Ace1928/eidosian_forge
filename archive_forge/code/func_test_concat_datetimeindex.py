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
def test_concat_datetimeindex():
    b2 = pd.DataFrame({'x': ['a']}, index=pd.DatetimeIndex(['2015-03-24 00:00:16'], dtype='datetime64[ns]'))
    b3 = pd.DataFrame({'x': ['c']}, index=pd.DatetimeIndex(['2015-03-29 00:00:44'], dtype='datetime64[ns]'))
    b2['x'] = b2.x.astype('category').cat.set_categories(['a', 'c'])
    b3['x'] = b3.x.astype('category').cat.set_categories(['a', 'c'])
    db2 = dd.from_pandas(b2, 1)
    db3 = dd.from_pandas(b3, 1)
    result = concat([b2.iloc[:0], b3.iloc[:0]])
    assert result.index.dtype == 'M8[ns]'
    result = dd.concat([db2, db3])
    expected = pd.concat([b2, b3])
    assert_eq(result, expected)