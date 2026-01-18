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
def test_groupby_cumsum_skipna_false():
    arr = np.random.default_rng(2).standard_normal((5, 5))
    df = DataFrame(arr)
    for i in range(5):
        df.iloc[i, i] = np.nan
    df['A'] = 1
    gb = df.groupby('A')
    res = gb.cumsum(skipna=False)
    expected = df[[0, 1, 2, 3, 4]].cumsum(skipna=False)
    tm.assert_frame_equal(res, expected)