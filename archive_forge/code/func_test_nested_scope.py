import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_nested_scope(self, engine, parser):
    x = 1
    result = pd.eval('x + 1', engine=engine, parser=parser)
    assert result == 2
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
    df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
    msg = "The '@' prefix is only supported by the pandas parser"
    with pytest.raises(SyntaxError, match=msg):
        df.query('(@df>0) & (@df2>0)', engine=engine, parser=parser)
    with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
        df.query('(df>0) & (df2>0)', engine=engine, parser=parser)
    expected = df[(df > 0) & (df2 > 0)]
    result = pd.eval('df[(df > 0) & (df2 > 0)]', engine=engine, parser=parser)
    tm.assert_frame_equal(expected, result)
    expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
    result = pd.eval('df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]', engine=engine, parser=parser)
    tm.assert_frame_equal(expected, result)