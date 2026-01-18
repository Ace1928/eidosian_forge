from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_agg_mapping_func_deprecated():
    df = DataFrame({'x': [1, 2, 3]})

    def foo1(x, a=1, c=0):
        return x + a + c

    def foo2(x, b=2, c=0):
        return x + b + c
    result = df.agg(foo1, 0, 3, c=4)
    expected = df + 7
    tm.assert_frame_equal(result, expected)
    msg = 'using .+ in Series.agg cannot aggregate and'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.agg([foo1, foo2], 0, 3, c=4)
    expected = DataFrame([[8, 8], [9, 9], [10, 10]], columns=[['x', 'x'], ['foo1', 'foo2']])
    tm.assert_frame_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.agg({'x': foo1}, 0, 3, c=4)
    expected = DataFrame([2, 3, 4], columns=['x'])
    tm.assert_frame_equal(result, expected)