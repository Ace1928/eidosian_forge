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
def test_merge_outer_empty():
    k_clusters = 3
    df = pd.DataFrame({'user': ['A', 'B', 'C', 'D', 'E', 'F'], 'cluster': [1, 1, 2, 2, 3, 3]})
    df = dd.from_pandas(df, npartitions=10)
    empty_df = dd.from_pandas(pd.DataFrame(), npartitions=10)
    for x in range(0, k_clusters + 1):
        assert_eq(dd.merge(empty_df, df[df.cluster == x], how='outer'), df[df.cluster == x], check_index=False, check_divisions=False)