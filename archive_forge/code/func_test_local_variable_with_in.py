import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_local_variable_with_in(self, engine, parser):
    skip_if_no_pandas_parser(parser)
    a = Series(np.random.default_rng(2).integers(3, size=15), name='a')
    b = Series(np.random.default_rng(2).integers(10, size=15), name='b')
    df = DataFrame({'a': a, 'b': b})
    expected = df.loc[(df.b - 1).isin(a)]
    result = df.query('b - 1 in a', engine=engine, parser=parser)
    tm.assert_frame_equal(expected, result)
    b = Series(np.random.default_rng(2).integers(10, size=15), name='b')
    expected = df.loc[(b - 1).isin(a)]
    result = df.query('@b - 1 in a', engine=engine, parser=parser)
    tm.assert_frame_equal(expected, result)