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
@pytest.mark.parametrize('other', [pd.timedelta_range('1D', periods=10), pd.timedelta_range('1D', periods=10).to_series(), pd.timedelta_range('1D', periods=10).asi8.view('m8[ns]')], ids=lambda x: type(x).__name__)
def test_dti_cmp_tdi_tzawareness(self, other):
    dti = date_range('2000-01-01', periods=10, tz='Asia/Tokyo')
    result = dti == other
    expected = np.array([False] * 10)
    tm.assert_numpy_array_equal(result, expected)
    result = dti != other
    expected = np.array([True] * 10)
    tm.assert_numpy_array_equal(result, expected)
    msg = 'Invalid comparison between'
    with pytest.raises(TypeError, match=msg):
        dti < other
    with pytest.raises(TypeError, match=msg):
        dti <= other
    with pytest.raises(TypeError, match=msg):
        dti > other
    with pytest.raises(TypeError, match=msg):
        dti >= other