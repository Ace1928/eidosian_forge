from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_crash_on_nunique(axis):
    dti = date_range('2016-01-01', periods=2, name='foo')
    df = DataFrame({('A', 'B'): [1, 2], ('A', 'C'): [1, 3], ('D', 'B'): [0, 0]})
    df.columns.names = ('bar', 'baz')
    df.index = dti
    axis_number = df._get_axis_number(axis)
    if not axis_number:
        df = df.T
        msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    else:
        msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(axis=axis_number, level=0)
    result = gb.nunique()
    expected = DataFrame({'A': [1, 2], 'D': [1, 1]}, index=dti)
    expected.columns.name = 'bar'
    if not axis_number:
        expected = expected.T
    tm.assert_frame_equal(result, expected)
    if axis_number == 0:
        with tm.assert_produces_warning(FutureWarning, match=msg):
            gb2 = df[[]].groupby(axis=axis_number, level=0)
        exp = expected[[]]
    else:
        with tm.assert_produces_warning(FutureWarning, match=msg):
            gb2 = df.loc[[]].groupby(axis=axis_number, level=0)
        exp = expected.loc[[]].astype(np.float64)
    res = gb2.nunique()
    tm.assert_frame_equal(res, exp)