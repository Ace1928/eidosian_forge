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
def test_dti_sub_tdi(self, tz_naive_fixture):
    tz = tz_naive_fixture
    dti = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10)
    tdi = pd.timedelta_range('0 days', periods=10)
    expected = date_range('2017-01-01', periods=10, tz=tz, freq='-1D')
    expected = expected._with_freq(None)
    result = dti - tdi
    tm.assert_index_equal(result, expected)
    msg = 'cannot subtract .*TimedeltaArray'
    with pytest.raises(TypeError, match=msg):
        tdi - dti
    result = dti - tdi.values
    tm.assert_index_equal(result, expected)
    msg = 'cannot subtract a datelike from a TimedeltaArray'
    with pytest.raises(TypeError, match=msg):
        tdi.values - dti