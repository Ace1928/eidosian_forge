from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_agg_list_like_func_with_args():
    df = DataFrame({'x': [1, 2, 3]}, index=date_range('2020-01-01', periods=3, freq='D'))

    def foo1(x, a=1, c=0):
        return x + a + c

    def foo2(x, b=2, c=0):
        return x + b + c
    msg = "foo1\\(\\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        df.resample('D').agg([foo1, foo2], 3, b=3, c=4)
    result = df.resample('D').agg([foo1, foo2], 3, c=4)
    expected = DataFrame([[8, 8], [9, 9], [10, 10]], index=date_range('2020-01-01', periods=3, freq='D'), columns=pd.MultiIndex.from_tuples([('x', 'foo1'), ('x', 'foo2')]))
    tm.assert_frame_equal(result, expected)