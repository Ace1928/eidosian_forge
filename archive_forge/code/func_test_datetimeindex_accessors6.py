import calendar
from datetime import datetime
import locale
import unicodedata
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.tseries.frequencies import to_offset
def test_datetimeindex_accessors6(self):
    dates = ['2013/12/29', '2013/12/30', '2013/12/31']
    dates = DatetimeIndex(dates, tz='Europe/Brussels')
    expected = [52, 1, 1]
    assert dates.isocalendar().week.tolist() == expected
    assert [d.weekofyear for d in dates] == expected