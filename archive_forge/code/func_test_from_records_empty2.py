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
def test_from_records_empty2(self):
    dtype = [('prop', int)]
    shape = (0, len(dtype))
    arr = np.empty(shape, dtype=dtype)
    result = DataFrame.from_records(arr)
    expected = DataFrame({'prop': np.array([], dtype=int)})
    tm.assert_frame_equal(result, expected)
    alt = DataFrame(arr)
    tm.assert_frame_equal(alt, expected)