from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
def test_groupby_rolling():
    df = pd.DataFrame({'column1': range(600), 'group1': 5 * ['g' + str(i) for i in range(120)]}, index=pd.date_range('20190101', periods=60).repeat(10))
    ddf = dd.from_pandas(df, npartitions=8)
    expected = df.groupby('group1').rolling('15D').sum()
    actual = ddf.groupby('group1').rolling('15D').sum()
    assert_eq(expected, actual, check_divisions=False)
    expected = df.groupby('group1').column1.rolling('15D').mean()
    actual = ddf.groupby('group1').column1.rolling('15D').mean()
    assert_eq(expected, actual, check_divisions=False)