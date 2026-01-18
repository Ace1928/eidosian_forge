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
@pytest.mark.arm_slow
def test_dt64arr_add_timedeltalike_scalar(self, tz_naive_fixture, two_hours, box_with_array):
    tz = tz_naive_fixture
    rng = date_range('2000-01-01', '2000-02-01', tz=tz)
    expected = date_range('2000-01-01 02:00', '2000-02-01 02:00', tz=tz)
    rng = tm.box_expected(rng, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    result = rng + two_hours
    tm.assert_equal(result, expected)
    result = two_hours + rng
    tm.assert_equal(result, expected)
    rng += two_hours
    tm.assert_equal(rng, expected)