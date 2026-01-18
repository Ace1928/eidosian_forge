from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_little_endian
from pandas import (
import pandas._testing as tm
def test_from_records_with_datetimes(self):
    if not is_platform_little_endian():
        pytest.skip('known failure of test on non-little endian')
    expected = DataFrame({'EXPIRY': [datetime(2005, 3, 1, 0, 0), None]})
    arrdata = [np.array([datetime(2005, 3, 1, 0, 0), None])]
    dtypes = [('EXPIRY', '<M8[ns]')]
    recarray = np.rec.fromarrays(arrdata, dtype=dtypes)
    result = DataFrame.from_records(recarray)
    tm.assert_frame_equal(result, expected)
    arrdata = [np.array([datetime(2005, 3, 1, 0, 0), None])]
    dtypes = [('EXPIRY', '<M8[m]')]
    recarray = np.rec.fromarrays(arrdata, dtype=dtypes)
    result = DataFrame.from_records(recarray)
    expected['EXPIRY'] = expected['EXPIRY'].astype('M8[s]')
    tm.assert_frame_equal(result, expected)