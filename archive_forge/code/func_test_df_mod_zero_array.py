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
def test_df_mod_zero_array(self):
    df = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]})
    first = Series([0, 0, 0, 0], dtype='float64')
    second = Series([np.nan, np.nan, np.nan, 0])
    expected = pd.DataFrame({'first': first, 'second': second})
    with np.errstate(all='ignore'):
        arr = df.values % df.values
    result2 = pd.DataFrame(arr, index=df.index, columns=df.columns, dtype='float64')
    result2.iloc[0:3, 1] = np.nan
    tm.assert_frame_equal(result2, expected)