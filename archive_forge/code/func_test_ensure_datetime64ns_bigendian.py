from datetime import datetime
import numpy as np
import pytest
from pytz import UTC
from pandas._libs.tslibs import (
from pandas import (
import pandas._testing as tm
def test_ensure_datetime64ns_bigendian():
    arr = np.array([np.datetime64(1, 'ms')], dtype='>M8[ms]')
    result = astype_overflowsafe(arr, dtype=np.dtype('M8[ns]'))
    expected = np.array([np.datetime64(1, 'ms')], dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)