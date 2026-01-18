from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_closed_uneven():
    ser = Series(data=np.arange(10), index=date_range('2000', periods=10))
    ser = ser.drop(index=ser.index[[1, 5]])
    result = ser.rolling('3D', closed='left').min()
    expected = Series([np.nan, 0, 0, 2, 3, 4, 6, 6], index=ser.index)
    tm.assert_series_equal(result, expected)