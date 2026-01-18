import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_chained_cmp_and_in(self, engine, parser):
    skip_if_no_pandas_parser(parser)
    cols = list('abc')
    df = DataFrame(np.random.default_rng(2).standard_normal((100, len(cols))), columns=cols)
    res = df.query('a < b < c and a not in b not in c', engine=engine, parser=parser)
    ind = (df.a < df.b) & (df.b < df.c) & ~df.b.isin(df.a) & ~df.c.isin(df.b)
    expec = df[ind]
    tm.assert_frame_equal(res, expec)