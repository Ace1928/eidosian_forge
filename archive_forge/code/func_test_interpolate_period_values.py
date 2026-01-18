import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(reason="EA.fillna does not handle 'linear' method")
def test_interpolate_period_values(self):
    orig = Series(date_range('2012-01-01', periods=5))
    ser = orig.copy()
    ser[2] = pd.NaT
    ser_per = ser.dt.to_period('D')
    res_per = ser_per.interpolate()
    expected_per = orig.dt.to_period('D')
    tm.assert_series_equal(res_per, expected_per)