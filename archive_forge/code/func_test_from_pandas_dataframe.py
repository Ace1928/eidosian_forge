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
def test_from_pandas_dataframe():
    a = list('aaaaaaabbbbbbbbccccccc')
    df = pd.DataFrame(dict(a=a, b=np.random.randn(len(a))), index=pd.date_range(start='20120101', periods=len(a)))
    ddf = dd.from_pandas(df, 3)
    expected_layers = 6 if pyarrow_strings_enabled() and (not DASK_EXPR_ENABLED) else 3
    assert len(ddf.dask) == expected_layers
    assert len(ddf.divisions) == 4
    assert isinstance(ddf.divisions[0], type(df.index[0]))
    assert_eq(df, ddf)
    ddf = dd.from_pandas(df, chunksize=8)
    msg = 'Exactly one of npartitions and chunksize must be specified.'
    with pytest.raises(ValueError) as err:
        dd.from_pandas(df, npartitions=2, chunksize=2)
    assert msg in str(err.value)
    if not DASK_EXPR_ENABLED:
        with pytest.raises((ValueError, AssertionError)) as err:
            dd.from_pandas(df)
        assert msg in str(err.value)
    assert len(ddf.dask) == expected_layers
    assert len(ddf.divisions) == 4
    assert isinstance(ddf.divisions[0], type(df.index[0]))
    assert_eq(df, ddf)