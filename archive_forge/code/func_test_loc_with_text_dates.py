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
def test_loc_with_text_dates():
    A = dd._compat.makeTimeSeries().iloc[:5]
    B = dd._compat.makeTimeSeries().iloc[5:]
    if DASK_EXPR_ENABLED:
        s = dd.repartition(pd.concat([A, B]), divisions=[A.index.min(), B.index.min(), B.index.max()])
    else:
        s = dd.Series({('df', 0): A, ('df', 1): B}, 'df', A, [A.index.min(), B.index.min(), B.index.max()])
    assert s.loc['2000':'2010'].divisions == s.divisions
    assert_eq(s.loc['2000':'2010'], s)
    assert len(s.loc['2000-01-03':'2000-01-05'].compute()) == 3