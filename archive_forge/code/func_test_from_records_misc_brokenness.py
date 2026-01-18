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
def test_from_records_misc_brokenness(self):
    data = {1: ['foo'], 2: ['bar']}
    result = DataFrame.from_records(data, columns=['a', 'b'])
    exp = DataFrame(data, columns=['a', 'b'])
    tm.assert_frame_equal(result, exp)
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    result = DataFrame.from_records(data, index=['a', 'b', 'c'])
    exp = DataFrame(data, index=['a', 'b', 'c'])
    tm.assert_frame_equal(result, exp)