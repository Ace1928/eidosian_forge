from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('closed', ['neither', 'left'])
def test_closed_empty(closed, arithmetic_win_operators):
    func_name = arithmetic_win_operators
    ser = Series(data=np.arange(5), index=date_range('2000', periods=5, freq='2D'))
    roll = ser.rolling('1D', closed=closed)
    result = getattr(roll, func_name)()
    expected = Series([np.nan] * 5, index=ser.index)
    tm.assert_series_equal(result, expected)