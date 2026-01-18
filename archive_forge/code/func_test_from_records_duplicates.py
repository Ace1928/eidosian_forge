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
def test_from_records_duplicates(self):
    result = DataFrame.from_records([(1, 2, 3), (4, 5, 6)], columns=['a', 'b', 'a'])
    expected = DataFrame([(1, 2, 3), (4, 5, 6)], columns=['a', 'b', 'a'])
    tm.assert_frame_equal(result, expected)