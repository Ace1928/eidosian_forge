from __future__ import annotations
import contextlib
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import tokenize
from dask.dataframe._compat import PANDAS_GE_210, PANDAS_GE_220, IndexingError, tm
from dask.dataframe.indexing import _coerce_loc_index
from dask.dataframe.utils import assert_eq, make_meta, pyarrow_strings_enabled
def test_loc_with_series_different_partition():
    df = pd.DataFrame(np.random.randn(20, 5), index=list('abcdefghijklmnopqrst'), columns=list('ABCDE'))
    ddf = dd.from_pandas(df, 3)
    assert_eq(ddf.loc[ddf.A > 0], df.loc[df.A > 0])
    assert_eq(ddf.loc[(ddf.A > 0).repartition(['a', 'g', 'k', 'o', 't'])], df.loc[df.A > 0])