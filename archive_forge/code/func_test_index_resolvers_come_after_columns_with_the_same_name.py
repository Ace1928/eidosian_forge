import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_index_resolvers_come_after_columns_with_the_same_name(self, engine, parser):
    n = 1
    a = np.r_[20:101:20]
    df = DataFrame({'index': a, 'b': np.random.default_rng(2).standard_normal(a.size)})
    df.index.name = 'index'
    result = df.query('index > 5', engine=engine, parser=parser)
    expected = df[df['index'] > 5]
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'index': a, 'b': np.random.default_rng(2).standard_normal(a.size)})
    result = df.query('ilevel_0 > 5', engine=engine, parser=parser)
    expected = df.loc[df.index[df.index > 5]]
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'a': a, 'b': np.random.default_rng(2).standard_normal(a.size)})
    df.index.name = 'a'
    result = df.query('a > 5', engine=engine, parser=parser)
    expected = df[df.a > 5]
    tm.assert_frame_equal(result, expected)
    result = df.query('index > 5', engine=engine, parser=parser)
    expected = df.loc[df.index[df.index > 5]]
    tm.assert_frame_equal(result, expected)