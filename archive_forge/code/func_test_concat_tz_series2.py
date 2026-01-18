import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_tz_series2(self):
    x = Series(date_range('20151124 08:00', '20151124 09:00', freq='1h', tz='UTC'))
    y = Series(['a', 'b'])
    expected = Series([x[0], x[1], y[0], y[1]], dtype='object')
    result = concat([x, y], ignore_index=True)
    tm.assert_series_equal(result, expected)