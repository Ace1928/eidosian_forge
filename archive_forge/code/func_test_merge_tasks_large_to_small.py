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
@pytest.mark.parametrize('how', ['inner', 'left', 'right'])
@pytest.mark.parametrize('npartitions', [28, 32])
@pytest.mark.parametrize('base', ['lg', 'sm'])
def test_merge_tasks_large_to_small(how, npartitions, base):
    size_lg = 3000
    size_sm = 300
    npartitions_lg = 30
    npartitions_sm = 3
    broadcast_bias = 1.0
    lg = pd.DataFrame({'x': np.random.choice(np.arange(100), size_lg), 'y': np.arange(size_lg)})
    ddf_lg = dd.from_pandas(lg, npartitions=npartitions_lg)
    sm = pd.DataFrame({'x': np.random.choice(np.arange(100), size_sm), 'y': np.arange(size_sm)})
    ddf_sm = dd.from_pandas(sm, npartitions=npartitions_sm)
    if base == 'lg':
        left = lg
        ddf_left = ddf_lg
        right = sm
        ddf_right = ddf_sm
    else:
        left = sm
        ddf_left = ddf_sm
        right = lg
        ddf_right = ddf_lg
    dd_result = dd.merge(ddf_left, ddf_right, on='y', how=how, npartitions=npartitions, broadcast=broadcast_bias, shuffle_method='tasks')
    pd_result = pd.merge(left, right, on='y', how=how)
    dd_result['y'] = dd_result['y'].astype(np.int32)
    pd_result['y'] = pd_result['y'].astype(np.int32)
    if npartitions:
        assert dd_result.npartitions == npartitions
    assert_eq(dd_result.compute().sort_values('y'), pd_result.sort_values('y'), check_index=False)