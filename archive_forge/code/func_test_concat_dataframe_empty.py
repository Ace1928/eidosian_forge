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
def test_concat_dataframe_empty():
    df = pd.DataFrame({'a': [100, 200, 300]}, dtype='int64')
    empty_df = pd.DataFrame([], dtype='int64')
    df_concat = pd.concat([df, empty_df])
    ddf = dd.from_pandas(df, npartitions=1)
    empty_ddf = dd.from_pandas(empty_df, npartitions=1)
    ddf_concat = dd.concat([ddf, empty_ddf])
    assert_eq(df_concat, ddf_concat)
    empty_df_with_col = pd.DataFrame([], columns=['x'], dtype='int64')
    df_concat_with_col = pd.concat([df, empty_df_with_col])
    empty_ddf_with_col = dd.from_pandas(empty_df_with_col, npartitions=1)
    ddf_concat_with_col = dd.concat([ddf, empty_ddf_with_col])
    ddf_concat_with_col._meta.index = ddf_concat_with_col._meta.index.astype('int64')
    assert_eq(df_concat_with_col, ddf_concat_with_col, check_dtype=False)
    assert_eq(dd.concat([ddf, ddf[[]]]), pd.concat([df, df[[]]]))