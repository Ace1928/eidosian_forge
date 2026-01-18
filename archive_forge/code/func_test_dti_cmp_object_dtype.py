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
def test_dti_cmp_object_dtype(self):
    dti = date_range('2000-01-01', periods=10, tz='Asia/Tokyo')
    other = dti.astype('O')
    result = dti == other
    expected = np.array([True] * 10)
    tm.assert_numpy_array_equal(result, expected)
    other = dti.tz_localize(None)
    result = dti != other
    tm.assert_numpy_array_equal(result, expected)
    other = np.array(list(dti[:5]) + [Timedelta(days=1)] * 5)
    result = dti == other
    expected = np.array([True] * 5 + [False] * 5)
    tm.assert_numpy_array_equal(result, expected)
    msg = ">=' not supported between instances of 'Timestamp' and 'Timedelta'"
    with pytest.raises(TypeError, match=msg):
        dti >= other