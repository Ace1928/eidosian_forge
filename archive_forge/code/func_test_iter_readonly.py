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
def test_iter_readonly():
    arr = np.array([np.datetime64('2012-02-15T12:00:00.000000000')])
    arr.setflags(write=False)
    dti = pd.to_datetime(arr)
    list(dti)