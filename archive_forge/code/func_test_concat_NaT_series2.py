import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_NaT_series2(self):
    x = Series(date_range('20151124 08:00', '20151124 09:00', freq='1h'))
    y = Series(date_range('20151124 10:00', '20151124 11:00', freq='1h'))
    y[:] = pd.NaT
    expected = Series([x[0], x[1], pd.NaT, pd.NaT])
    result = concat([x, y], ignore_index=True)
    tm.assert_series_equal(result, expected)
    x[:] = pd.NaT
    expected = Series(pd.NaT, index=range(4), dtype='datetime64[ns]')
    result = concat([x, y], ignore_index=True)
    tm.assert_series_equal(result, expected)