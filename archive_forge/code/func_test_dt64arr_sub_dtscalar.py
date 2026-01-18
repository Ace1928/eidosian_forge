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
@pytest.mark.parametrize('ts', [Timestamp('2013-01-01'), Timestamp('2013-01-01').to_pydatetime(), Timestamp('2013-01-01').to_datetime64(), np.datetime64('2013-01-01', 'D')])
def test_dt64arr_sub_dtscalar(self, box_with_array, ts):
    idx = date_range('2013-01-01', periods=3)._with_freq(None)
    idx = tm.box_expected(idx, box_with_array)
    expected = TimedeltaIndex(['0 Days', '1 Day', '2 Days'])
    expected = tm.box_expected(expected, box_with_array)
    result = idx - ts
    tm.assert_equal(result, expected)
    result = ts - idx
    tm.assert_equal(result, -expected)
    tm.assert_equal(result, -expected)