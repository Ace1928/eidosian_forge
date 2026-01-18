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
@pytest.mark.parametrize('other_box', [pd.Index, np.array])
def test_dti_addsub_object_arraylike(self, tz_naive_fixture, box_with_array, other_box):
    tz = tz_naive_fixture
    dti = date_range('2017-01-01', periods=2, tz=tz)
    dtarr = tm.box_expected(dti, box_with_array)
    other = other_box([pd.offsets.MonthEnd(), Timedelta(days=4)])
    xbox = get_upcast_box(dtarr, other)
    expected = DatetimeIndex(['2017-01-31', '2017-01-06'], tz=tz_naive_fixture)
    expected = tm.box_expected(expected, xbox).astype(object)
    with tm.assert_produces_warning(PerformanceWarning):
        result = dtarr + other
    tm.assert_equal(result, expected)
    expected = DatetimeIndex(['2016-12-31', '2016-12-29'], tz=tz_naive_fixture)
    expected = tm.box_expected(expected, xbox).astype(object)
    with tm.assert_produces_warning(PerformanceWarning):
        result = dtarr - other
    tm.assert_equal(result, expected)