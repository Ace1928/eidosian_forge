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
@pytest.mark.skipif(PANDAS_GE_200, reason='pandas removed append')
def test_append_lose_divisions():
    df = pd.DataFrame({'x': [1, 2, 3, 4]}, index=[1, 2, 3, 4])
    ddf = dd.from_pandas(df, npartitions=2)
    res = check_append_with_warning(ddf, ddf, df, df)
    assert res.known_divisions is False