from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.strptime import array_strptime
from pandas import (
import pandas._testing as tm
def test_array_strptime_str_outside_nano_range(self):
    vals = np.array(['2401-09-15'], dtype=object)
    expected = np.array(['2401-09-15'], dtype='M8[s]')
    fmt = 'ISO8601'
    res, _ = array_strptime(vals, fmt=fmt, creso=creso_infer)
    tm.assert_numpy_array_equal(res, expected)
    vals2 = np.array(['Sep 15, 2401'], dtype=object)
    expected2 = np.array(['2401-09-15'], dtype='M8[s]')
    fmt2 = '%b %d, %Y'
    res2, _ = array_strptime(vals2, fmt=fmt2, creso=creso_infer)
    tm.assert_numpy_array_equal(res2, expected2)