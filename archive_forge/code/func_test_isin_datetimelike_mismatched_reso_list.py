import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray
def test_isin_datetimelike_mismatched_reso_list(self):
    expected = Series([True, True, False, False, False])
    ser = Series(date_range('jan-01-2013', 'jan-05-2013'))
    dta = ser[:2]._values.astype('M8[s]')
    result = ser.isin(list(dta))
    tm.assert_series_equal(result, expected)