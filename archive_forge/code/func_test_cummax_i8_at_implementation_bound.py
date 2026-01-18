import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
def test_cummax_i8_at_implementation_bound():
    ser = Series([pd.NaT._value + n for n in range(5)])
    df = DataFrame({'A': 1, 'B': ser, 'C': ser.view('M8[ns]')})
    gb = df.groupby('A')
    res = gb.cummax()
    exp = df[['B', 'C']]
    tm.assert_frame_equal(res, exp)