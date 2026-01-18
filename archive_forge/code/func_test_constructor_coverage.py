from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_constructor_coverage(self):
    msg = 'DatetimeIndex\\(\\.\\.\\.\\) must be called with a collection'
    with pytest.raises(TypeError, match=msg):
        DatetimeIndex('1/1/2000')
    gen = (datetime(2000, 1, 1) + timedelta(i) for i in range(10))
    result = DatetimeIndex(gen)
    expected = DatetimeIndex([datetime(2000, 1, 1) + timedelta(i) for i in range(10)])
    tm.assert_index_equal(result, expected)
    strings = np.array(['2000-01-01', '2000-01-02', '2000-01-03'])
    result = DatetimeIndex(strings)
    expected = DatetimeIndex(strings.astype('O'))
    tm.assert_index_equal(result, expected)
    from_ints = DatetimeIndex(expected.asi8)
    tm.assert_index_equal(from_ints, expected)
    strings = np.array(['2000-01-01', '2000-01-02', 'NaT'])
    result = DatetimeIndex(strings)
    expected = DatetimeIndex(strings.astype('O'))
    tm.assert_index_equal(result, expected)
    from_ints = DatetimeIndex(expected.asi8)
    tm.assert_index_equal(from_ints, expected)
    msg = 'Inferred frequency None from passed values does not conform to passed frequency D'
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex(['2000-01-01', '2000-01-02', '2000-01-04'], freq='D')