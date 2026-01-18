from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
def test_df_div_zero_int(self):
    df = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]})
    result = df / 0
    expected = pd.DataFrame(np.inf, index=df.index, columns=df.columns)
    expected.iloc[0:3, 1] = np.nan
    tm.assert_frame_equal(result, expected)
    with np.errstate(all='ignore'):
        arr = df.values.astype('float64') / 0
    result2 = pd.DataFrame(arr, index=df.index, columns=df.columns)
    tm.assert_frame_equal(result2, expected)