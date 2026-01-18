from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_constructor_timedelta64_values_mismatched_dtype(self):
    tdi = timedelta_range('4 Days', periods=5)
    result = Index(tdi, dtype='category')
    expected = CategoricalIndex(tdi)
    tm.assert_index_equal(result, expected)