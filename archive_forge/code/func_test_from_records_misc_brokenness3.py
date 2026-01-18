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
def test_from_records_misc_brokenness3(self):
    rows = []
    rows.append([datetime(2010, 1, 1), 1])
    rows.append([datetime(2010, 1, 2), 1])
    result = DataFrame.from_records(rows, columns=['date', 'test'])
    expected = DataFrame({'date': [row[0] for row in rows], 'test': [row[1] for row in rows]})
    tm.assert_frame_equal(result, expected)