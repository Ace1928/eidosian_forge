from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
def test_coerce_outside_ns_bounds_one_valid():
    arr = np.array(['1/1/1000', '1/1/2000'], dtype=object)
    result, _ = tslib.array_to_datetime(arr, errors='coerce')
    expected = [iNaT, '2000-01-01T00:00:00.000000000']
    expected = np.array(expected, dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)