from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_constructor_no_pandas_array(self):
    ser = Series([1, 2, 3])
    result = Index(ser.array)
    expected = Index([1, 2, 3])
    tm.assert_index_equal(result, expected)