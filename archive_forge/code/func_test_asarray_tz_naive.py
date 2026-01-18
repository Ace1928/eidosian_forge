import datetime as dt
from datetime import date
import re
import numpy as np
import pytest
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_asarray_tz_naive(self):
    idx = date_range('2000', periods=2)
    result = np.asarray(idx)
    expected = np.array(['2000-01-01', '2000-01-02'], dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)
    result = np.asarray(idx, dtype=object)
    expected = np.array([Timestamp('2000-01-01'), Timestamp('2000-01-02')])
    tm.assert_numpy_array_equal(result, expected)