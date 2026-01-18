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
def test_groupby_cumsum_timedelta64():
    dti = date_range('2016-01-01', periods=5)
    ser = Series(dti) - dti[0]
    ser[2] = pd.NaT
    df = DataFrame({'A': 1, 'B': ser})
    gb = df.groupby('A')
    res = gb.cumsum(numeric_only=False, skipna=True)
    exp = DataFrame({'B': [ser[0], ser[1], pd.NaT, ser[4], ser[4] * 2]})
    tm.assert_frame_equal(res, exp)
    res = gb.cumsum(numeric_only=False, skipna=False)
    exp = DataFrame({'B': [ser[0], ser[1], pd.NaT, pd.NaT, pd.NaT]})
    tm.assert_frame_equal(res, exp)