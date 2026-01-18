from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.base import (
@pytest.mark.parametrize('method', ['count', 'corr', 'cummax', 'cummin', 'cumprod', 'describe', 'rank', 'quantile', 'diff', 'shift', 'all', 'any', 'idxmin', 'idxmax', 'ffill', 'bfill', 'pct_change'])
def test_groupby_selection_with_methods(df, method):
    rng = date_range('2014', periods=len(df))
    df.index = rng
    g = df.groupby(['A'])[['C']]
    g_exp = df[['C']].groupby(df['A'])
    res = getattr(g, method)()
    exp = getattr(g_exp, method)()
    tm.assert_frame_equal(res, exp)