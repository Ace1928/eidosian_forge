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
def test_from_records_empty_with_nonempty_fields_gh3682(self):
    a = np.array([(1, 2)], dtype=[('id', np.int64), ('value', np.int64)])
    df = DataFrame.from_records(a, index='id')
    ex_index = Index([1], name='id')
    expected = DataFrame({'value': [2]}, index=ex_index, columns=['value'])
    tm.assert_frame_equal(df, expected)
    b = a[:0]
    df2 = DataFrame.from_records(b, index='id')
    tm.assert_frame_equal(df2, df.iloc[:0])