from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def test_array_ufunc_series_defer():
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)
    expected = pd.Series(to_decimal([2, 4, 6]))
    r1 = np.add(s, a)
    r2 = np.add(a, s)
    tm.assert_series_equal(r1, expected)
    tm.assert_series_equal(r2, expected)