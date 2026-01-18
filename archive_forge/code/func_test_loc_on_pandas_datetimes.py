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
def test_loc_on_pandas_datetimes():
    df = pd.DataFrame({'x': [1, 2, 3]}, index=list(map(pd.Timestamp, ['2014', '2015', '2016'])))
    a = dd.from_pandas(df, 2)
    a = a.repartition(divisions=tuple(map(pd.Timestamp, a.divisions)))
    assert_eq(a.loc['2014':'2015'], a.loc['2014':'2015'])