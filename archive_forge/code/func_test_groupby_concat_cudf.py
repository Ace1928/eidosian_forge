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
@pytest.mark.parametrize('engine', ['pandas', pytest.param('cudf', marks=pytest.mark.gpu)])
def test_groupby_concat_cudf(engine):
    size = 6
    npartitions = 3
    d1 = pd.DataFrame({'a': np.random.permutation(np.arange(size)), 'b': np.random.randint(100, size=size)})
    d2 = pd.DataFrame({'c': np.random.permutation(np.arange(size)), 'd': np.random.randint(100, size=size)})
    if engine == 'cudf':
        cudf = pytest.importorskip('cudf')
        dask_cudf = pytest.importorskip('dask_cudf')
        d1 = cudf.from_pandas(d1)
        d2 = cudf.from_pandas(d2)
        dd1 = dask_cudf.from_cudf(d1, npartitions)
        dd2 = dask_cudf.from_cudf(d2, npartitions)
    else:
        dd1 = dd.from_pandas(d1, npartitions)
        dd2 = dd.from_pandas(d2, npartitions)
    grouped_d1 = d1.groupby(['a']).sum()
    grouped_d2 = d2.groupby(['c']).sum()
    res = concat([grouped_d1, grouped_d2], axis=1)
    grouped_dd1 = dd1.groupby(['a']).sum()
    grouped_dd2 = dd2.groupby(['c']).sum()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        res_dd = dd.concat([grouped_dd1, grouped_dd2], axis=1)
    assert_eq(res_dd.compute().sort_index(), res.sort_index())