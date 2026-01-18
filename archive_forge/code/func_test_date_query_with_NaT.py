import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_date_query_with_NaT(self, engine, parser):
    n = 10
    df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
    df['dates1'] = date_range('1/1/2012', periods=n)
    df['dates2'] = date_range('1/1/2013', periods=n)
    df['dates3'] = date_range('1/1/2014', periods=n)
    df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates1'] = pd.NaT
    df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates3'] = pd.NaT
    res = df.query('(dates1 < 20130101) & (20130101 < dates3)', engine=engine, parser=parser)
    expec = df[(df.dates1 < '20130101') & ('20130101' < df.dates3)]
    tm.assert_frame_equal(res, expec)