from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_partial_str_slice_with_timedeltaindex(self):
    rng = timedelta_range('1 day 10:11:12', freq='h', periods=500)
    ser = Series(np.arange(len(rng)), index=rng)
    result = ser['5 day':'6 day']
    expected = ser.iloc[86:134]
    tm.assert_series_equal(result, expected)
    result = ser['5 day':]
    expected = ser.iloc[86:]
    tm.assert_series_equal(result, expected)
    result = ser[:'6 day']
    expected = ser.iloc[:134]
    tm.assert_series_equal(result, expected)