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
def test_datetimeindex_accessors4(self):
    dti = DatetimeIndex(['2000-01-01', '2000-01-02', '2000-01-03'])
    assert dti.is_month_start[0] == 1