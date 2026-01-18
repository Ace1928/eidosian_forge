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
def test_concat_categorical_mixed_simple():
    a = pd.Series(['a', 'b', 'c'], dtype='category')
    b = pd.Series(['a', 'b'], dtype='category')
    da = dd.from_pandas(a, 2).cat.as_unknown().to_frame('A')
    db = dd.from_pandas(b, 2).to_frame('A')
    expected = concat([a.to_frame('A'), b.to_frame('A')])
    result = dd.concat([da, db])
    assert_eq(result, expected)