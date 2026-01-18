from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import dask
import dask.array as da
import dask.dataframe as dd
from dask import config
from dask.blockwise import Blockwise
from dask.dataframe._compat import PANDAS_GE_200, tm
from dask.dataframe.io.io import _meta_from_array
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import assert_eq, get_string_dtype, pyarrow_strings_enabled
from dask.delayed import Delayed, delayed
from dask.utils_test import hlg_layer_topological
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="doesn't make sense")
def test_from_delayed_preserves_hlgs():
    df = pd.DataFrame(data=np.random.normal(size=(10, 4)), columns=list('abcd'))
    parts = [df.iloc[:1], df.iloc[1:3], df.iloc[3:6], df.iloc[6:10]]
    dfs = [delayed(parts.__getitem__)(i) for i in range(4)]
    meta = dfs[0].compute()
    chained = [d.a for d in dfs]
    hlg = dd.from_delayed(chained, meta=meta).dask
    for d in chained:
        for layer_name, layer in d.dask.layers.items():
            assert hlg.layers[layer_name] == layer
            assert hlg.dependencies[layer_name] == d.dask.dependencies[layer_name]