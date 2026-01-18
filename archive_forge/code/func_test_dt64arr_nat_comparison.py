from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_dt64arr_nat_comparison(self, tz_naive_fixture, box_with_array):
    tz = tz_naive_fixture
    box = box_with_array
    ts = Timestamp('2021-01-01', tz=tz)
    ser = Series([ts, NaT])
    obj = tm.box_expected(ser, box)
    xbox = get_upcast_box(obj, ts, True)
    expected = Series([True, False], dtype=np.bool_)
    expected = tm.box_expected(expected, xbox)
    result = obj == ts
    tm.assert_equal(result, expected)