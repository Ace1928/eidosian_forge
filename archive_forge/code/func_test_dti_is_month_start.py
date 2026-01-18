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
def test_dti_is_month_start(self):
    dti = DatetimeIndex(['2000-01-01', '2000-01-02', '2000-01-03'])
    assert dti.is_month_start[0] == 1