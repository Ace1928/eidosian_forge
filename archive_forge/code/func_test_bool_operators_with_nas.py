from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.filterwarnings('ignore:Downcasting object dtype arrays:FutureWarning')
@pytest.mark.parametrize('bool_op', [operator.and_, operator.or_, operator.xor])
def test_bool_operators_with_nas(self, bool_op):
    ser = Series(bdate_range('1/1/2000', periods=10), dtype=object)
    ser[::2] = np.nan
    mask = ser.isna()
    filled = ser.fillna(ser[0])
    result = bool_op(ser < ser[9], ser > ser[3])
    expected = bool_op(filled < filled[9], filled > filled[3])
    expected[mask] = False
    tm.assert_series_equal(result, expected)