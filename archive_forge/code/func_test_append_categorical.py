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
def test_append_categorical():
    frames = [pd.DataFrame({'x': np.arange(5, 10), 'y': list('abbba'), 'z': np.arange(5, 10, dtype='f8')}), pd.DataFrame({'x': np.arange(10, 15), 'y': list('bcbcc'), 'z': np.arange(10, 15, dtype='f8')})]
    frames2 = []
    for df in frames:
        df.y = df.y.astype('category')
        df2 = df.copy()
        df2.y = df2.y.cat.set_categories(list('abc'))
        df.index = df.y
        frames2.append(df2.set_index(df2.y))
    df1, df2 = frames2
    for known in [True, False]:
        dframes = [dd.from_pandas(p, npartitions=2, sort=False) for p in frames]
        if not known:
            dframes[0]._meta = clear_known_categories(dframes[0]._meta, ['y'], index=True)
        ddf1, ddf2 = dframes
        res = check_append_with_warning(ddf1, ddf2, df1, df2)
        assert has_known_categories(res.index) == known
        assert has_known_categories(res.y) == known
        res = check_append_with_warning(ddf1.y, ddf2.y, df1.y, df2.y)
        assert has_known_categories(res.index) == known
        assert has_known_categories(res) == known
        res = check_append_with_warning(ddf1.index, ddf2.index, df1.index, df2.index)
        assert has_known_categories(res) == known