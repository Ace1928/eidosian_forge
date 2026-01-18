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
def test_dti_nanosecond(self):
    dti = DatetimeIndex(np.arange(10))
    expected = Index(np.arange(10, dtype=np.int32))
    tm.assert_index_equal(dti.nanosecond, expected)