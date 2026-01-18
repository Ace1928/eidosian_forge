import datetime as dt
from datetime import date
import re
import numpy as np
import pytest
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_CBH_deprecated(self):
    msg = "'CBH' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = date_range(dt.datetime(2022, 12, 11), dt.datetime(2022, 12, 13), freq='CBH')
    result = DatetimeIndex(['2022-12-12 09:00:00', '2022-12-12 10:00:00', '2022-12-12 11:00:00', '2022-12-12 12:00:00', '2022-12-12 13:00:00', '2022-12-12 14:00:00', '2022-12-12 15:00:00', '2022-12-12 16:00:00'], dtype='datetime64[ns]', freq='cbh')
    tm.assert_index_equal(result, expected)