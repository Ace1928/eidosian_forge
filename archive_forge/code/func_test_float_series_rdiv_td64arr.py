from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_float_series_rdiv_td64arr(self, box_with_array, names):
    box = box_with_array
    tdi = TimedeltaIndex(['0days', '1day', '2days', '3days', '4days'], name=names[0])
    ser = Series([1.5, 3, 4.5, 6, 7.5], dtype=np.float64, name=names[1])
    xname = names[2] if box not in [tm.to_array, pd.array] else names[1]
    expected = Series([tdi[n] / ser[n] for n in range(len(ser))], dtype='timedelta64[ns]', name=xname)
    tdi = tm.box_expected(tdi, box)
    xbox = get_upcast_box(tdi, ser)
    expected = tm.box_expected(expected, xbox)
    result = ser.__rtruediv__(tdi)
    if box is DataFrame:
        assert result is NotImplemented
    else:
        tm.assert_equal(result, expected)