import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_date_query_no_attribute_access(self, engine, parser):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
    df['dates1'] = date_range('1/1/2012', periods=5)
    df['dates2'] = date_range('1/1/2013', periods=5)
    df['dates3'] = date_range('1/1/2014', periods=5)
    res = df.query('(dates1 < 20130101) & (20130101 < dates3)', engine=engine, parser=parser)
    expec = df[(df.dates1 < '20130101') & ('20130101' < df.dates3)]
    tm.assert_frame_equal(res, expec)