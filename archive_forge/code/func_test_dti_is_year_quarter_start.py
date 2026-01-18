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
def test_dti_is_year_quarter_start(self):
    dti = date_range(freq='BQE-FEB', start=datetime(1998, 1, 1), periods=4)
    assert sum(dti.is_quarter_start) == 0
    assert sum(dti.is_quarter_end) == 4
    assert sum(dti.is_year_start) == 0
    assert sum(dti.is_year_end) == 1