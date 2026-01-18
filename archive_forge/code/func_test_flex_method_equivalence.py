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
@pytest.mark.parametrize('ts', [(lambda x: x, lambda x: x * 2, False), (lambda x: x, lambda x: x[::2], False), (lambda x: x, lambda x: 5, True), (lambda x: Series(range(10), dtype=np.float64), lambda x: Series(range(10), dtype=np.float64), True)])
@pytest.mark.parametrize('opname', ['add', 'sub', 'mul', 'floordiv', 'truediv', 'pow'])
def test_flex_method_equivalence(self, opname, ts):
    tser = Series(np.arange(20, dtype=np.float64), index=date_range('2020-01-01', periods=20), name='ts')
    series = ts[0](tser)
    other = ts[1](tser)
    check_reverse = ts[2]
    op = getattr(Series, opname)
    alt = getattr(operator, opname)
    result = op(series, other)
    expected = alt(series, other)
    tm.assert_almost_equal(result, expected)
    if check_reverse:
        rop = getattr(Series, 'r' + opname)
        result = rop(series, other)
        expected = alt(other, series)
        tm.assert_almost_equal(result, expected)