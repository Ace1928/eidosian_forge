from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def test_array_ufunc_series():
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)
    result = np.exp(s)
    expected = pd.Series(to_decimal(np.exp(a._data)))
    tm.assert_series_equal(result, expected)