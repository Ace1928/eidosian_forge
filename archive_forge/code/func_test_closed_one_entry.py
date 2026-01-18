from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('func', ['min', 'max'])
def test_closed_one_entry(func):
    ser = Series(data=[2], index=date_range('2000', periods=1))
    result = getattr(ser.rolling('10D', closed='left'), func)()
    tm.assert_series_equal(result, Series([np.nan], index=ser.index))