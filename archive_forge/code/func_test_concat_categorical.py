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
@pytest.mark.parametrize('known, cat_index, divisions', [(True, True, False), pytest.param(True, False, True, marks=pytest.mark.xfail(PANDAS_GE_220 or PY_VERSION >= Version('3.12.0'), reason='fails on pandas dev: https://github.com/dask/dask/issues/10558', raises=AssertionError, strict=False)), (True, False, False), (False, True, False), pytest.param(False, False, True, marks=pytest.mark.xfail(PANDAS_GE_220 or PY_VERSION >= Version('3.12.0'), reason='fails on pandas dev: https://github.com/dask/dask/issues/10558', raises=AssertionError, strict=False)), (False, False, False)])
def test_concat_categorical(known, cat_index, divisions):
    frames = [pd.DataFrame({'w': list('xxxxx'), 'x': np.arange(5), 'y': list('abcbc'), 'z': np.arange(5, dtype='f8')}), pd.DataFrame({'w': list('yyyyy'), 'x': np.arange(5, 10), 'y': list('abbba'), 'z': np.arange(5, 10, dtype='f8')}), pd.DataFrame({'w': list('zzzzz'), 'x': np.arange(10, 15), 'y': list('bcbcc'), 'z': np.arange(10, 15, dtype='f8')})]
    for df in frames:
        df.w = df.w.astype('category')
        df.y = df.y.astype('category')
    if cat_index:
        frames = [df.set_index(df.y) for df in frames]
    dframes = [dd.from_pandas(p, npartitions=2, sort=divisions) for p in frames]
    if not known:
        if DASK_EXPR_ENABLED:
            dframes[0]['y'] = dframes[0]['y'].cat.as_unknown()
            if cat_index:
                dframes[0].index = dframes[0].index.cat.as_unknown()
        else:
            dframes[0]._meta = clear_known_categories(dframes[0]._meta, ['y'], index=True)

    def check_and_return(ddfs, dfs, join):
        sol = concat(dfs, join=join)
        res = dd.concat(ddfs, join=join, interleave_partitions=divisions)
        assert_eq(res, sol)
        if known:
            parts = compute_as_if_collection(dd.DataFrame, res.dask, res.__dask_keys__())
            for p in [i.iloc[:0] for i in parts]:
                check_meta(res._meta, p)
        assert not cat_index or has_known_categories(res.index) == known
        return res
    for join in ['inner', 'outer']:
        res = check_and_return(dframes, frames, join)
        assert has_known_categories(res.w)
        assert has_known_categories(res.y) == known
        res = check_and_return([i.y for i in dframes], [i.y for i in frames], join)
        assert has_known_categories(res) == known
        if cat_index:
            res = check_and_return([i.x for i in dframes], [i.x for i in frames], join)
        res = check_and_return([dframes[0][['x', 'y']]] + dframes[1:], [frames[0][['x', 'y']]] + frames[1:], join)
        assert not hasattr(res, 'w') or has_known_categories(res.w)
        assert has_known_categories(res.y) == known