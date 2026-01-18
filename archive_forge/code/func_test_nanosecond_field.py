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
def test_nanosecond_field(self):
    dti = DatetimeIndex(np.arange(10))
    expected = Index(np.arange(10, dtype=np.int32))
    tm.assert_index_equal(dti.nanosecond, expected)