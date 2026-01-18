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
@pytest.mark.parametrize('tz', [None, 'UTC', 'US/Pacific'])
def test_dti_constructor_with_non_nano_dtype(self, tz):
    ts = Timestamp('2999-01-01')
    dtype = 'M8[us]'
    if tz is not None:
        dtype = f'M8[us, {tz}]'
    vals = [ts, '2999-01-02 03:04:05.678910', 2500]
    result = DatetimeIndex(vals, dtype=dtype)
    pointwise = [vals[0].tz_localize(tz), Timestamp(vals[1], tz=tz), to_datetime(vals[2], unit='us', utc=True).tz_convert(tz)]
    exp_vals = [x.as_unit('us').asm8 for x in pointwise]
    exp_arr = np.array(exp_vals, dtype='M8[us]')
    expected = DatetimeIndex(exp_arr, dtype='M8[us]')
    if tz is not None:
        expected = expected.tz_localize('UTC').tz_convert(tz)
    tm.assert_index_equal(result, expected)
    result2 = DatetimeIndex(np.array(vals, dtype=object), dtype=dtype)
    tm.assert_index_equal(result2, expected)