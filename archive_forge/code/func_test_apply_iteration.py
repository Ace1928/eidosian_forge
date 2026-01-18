from datetime import datetime
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
def test_apply_iteration():
    N = 1000
    ind = date_range(start='2000-01-01', freq='D', periods=N)
    df = DataFrame({'open': 1, 'close': 2}, index=ind)
    tg = Grouper(freq='ME')
    grouper, _ = tg._get_grouper(df)
    grouped = df.groupby(grouper, group_keys=False)

    def f(df):
        return df['close'] / df['open']
    result = grouped.apply(f)
    tm.assert_index_equal(result.index, df.index)