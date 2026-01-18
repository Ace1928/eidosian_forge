import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_dti_is_month_start_custom(self):
    bday_egypt = offsets.CustomBusinessDay(weekmask='Sun Mon Tue Wed Thu')
    dti = date_range(datetime(2013, 4, 30), periods=5, freq=bday_egypt)
    msg = 'Custom business days is not supported by is_month_start'
    with pytest.raises(ValueError, match=msg):
        dti.is_month_start