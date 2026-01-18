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
def test_datetimeindex_accessors3(self):
    bday_egypt = offsets.CustomBusinessDay(weekmask='Sun Mon Tue Wed Thu')
    dti = date_range(datetime(2013, 4, 30), periods=5, freq=bday_egypt)
    msg = 'Custom business days is not supported by is_month_start'
    with pytest.raises(ValueError, match=msg):
        dti.is_month_start