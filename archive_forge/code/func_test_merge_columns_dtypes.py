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
@pytest.mark.parametrize('how', ['inner', 'outer', 'left', 'right'])
@pytest.mark.parametrize('on_index', [True, False])
def test_merge_columns_dtypes(how, on_index):
    df1 = pd.DataFrame({'A': list(np.arange(5).astype(float)) * 2, 'B': list(np.arange(5)) * 2})
    df2 = pd.DataFrame({'A': np.arange(5), 'B': np.arange(5)})
    a = dd.from_pandas(df1, 2)
    b = dd.from_pandas(df2, 2)
    on = ['A']
    left_index = right_index = on_index
    if on_index:
        a = a.set_index('A')
        b = b.set_index('A')
        on = None
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        result = dd.merge(a, b, on=on, how=how, left_index=left_index, right_index=right_index)
        warned = any(('merge column data type mismatches' in str(r) for r in record))
    result = result if isinstance(result, pd.DataFrame) else result.compute()
    has_nans = result.isna().values.any()
    assert has_nans and warned or not has_nans