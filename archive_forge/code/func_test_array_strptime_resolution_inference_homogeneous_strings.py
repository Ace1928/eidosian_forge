from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.strptime import array_strptime
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, timezone.utc])
def test_array_strptime_resolution_inference_homogeneous_strings(self, tz):
    dt = datetime(2016, 1, 2, 3, 4, 5, 678900, tzinfo=tz)
    fmt = '%Y-%m-%d %H:%M:%S'
    dtstr = dt.strftime(fmt)
    arr = np.array([dtstr] * 3, dtype=object)
    expected = np.array([dt.replace(tzinfo=None)] * 3, dtype='M8[s]')
    res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
    tm.assert_numpy_array_equal(res, expected)
    fmt = '%Y-%m-%d %H:%M:%S.%f'
    dtstr = dt.strftime(fmt)
    arr = np.array([dtstr] * 3, dtype=object)
    expected = np.array([dt.replace(tzinfo=None)] * 3, dtype='M8[us]')
    res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
    tm.assert_numpy_array_equal(res, expected)
    fmt = 'ISO8601'
    res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
    tm.assert_numpy_array_equal(res, expected)