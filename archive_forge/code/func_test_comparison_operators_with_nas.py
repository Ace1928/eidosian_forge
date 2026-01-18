from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_comparison_operators_with_nas(self, comparison_op):
    ser = Series(bdate_range('1/1/2000', periods=10), dtype=object)
    ser[::2] = np.nan
    val = ser[5]
    result = comparison_op(ser, val)
    expected = comparison_op(ser.dropna(), val).reindex(ser.index)
    msg = 'Downcasting object dtype arrays'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        if comparison_op is operator.ne:
            expected = expected.fillna(True).astype(bool)
        else:
            expected = expected.fillna(False).astype(bool)
    tm.assert_series_equal(result, expected)