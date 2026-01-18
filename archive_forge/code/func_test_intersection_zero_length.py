import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import Hour
@pytest.mark.parametrize('period_1, period_2', [(0, 4), (4, 0)])
def test_intersection_zero_length(self, period_1, period_2, sort):
    index_1 = timedelta_range('1 day', periods=period_1, freq='h')
    index_2 = timedelta_range('1 day', periods=period_2, freq='h')
    expected = timedelta_range('1 day', periods=0, freq='h')
    result = index_1.intersection(index_2, sort=sort)
    tm.assert_index_equal(result, expected)