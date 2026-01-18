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
def test_groupby_reduce_period():
    pi = pd.period_range('2016-01-01', periods=100, freq='D')
    grps = list(range(10)) * 10
    ser = pi.to_series()
    gb = ser.groupby(grps)
    with pytest.raises(TypeError, match='Period type does not support sum operations'):
        gb.sum()
    with pytest.raises(TypeError, match='Period type does not support cumsum operations'):
        gb.cumsum()
    with pytest.raises(TypeError, match='Period type does not support prod operations'):
        gb.prod()
    with pytest.raises(TypeError, match='Period type does not support cumprod operations'):
        gb.cumprod()
    res = gb.max()
    expected = ser[-10:]
    expected.index = Index(range(10), dtype=int)
    tm.assert_series_equal(res, expected)
    res = gb.min()
    expected = ser[:10]
    expected.index = Index(range(10), dtype=int)
    tm.assert_series_equal(res, expected)