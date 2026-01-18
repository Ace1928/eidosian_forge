import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_NaT_series(self):
    x = Series(date_range('20151124 08:00', '20151124 09:00', freq='1h', tz='US/Eastern'))
    y = Series(pd.NaT, index=[0, 1], dtype='datetime64[ns, US/Eastern]')
    expected = Series([x[0], x[1], pd.NaT, pd.NaT])
    result = concat([x, y], ignore_index=True)
    tm.assert_series_equal(result, expected)
    expected = Series(pd.NaT, index=range(4), dtype='datetime64[ns, US/Eastern]')
    result = concat([y, y], ignore_index=True)
    tm.assert_series_equal(result, expected)