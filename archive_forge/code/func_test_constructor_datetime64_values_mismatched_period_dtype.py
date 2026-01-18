from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_constructor_datetime64_values_mismatched_period_dtype(self):
    dti = date_range('2016-01-01', periods=3)
    result = Index(dti, dtype='Period[D]')
    expected = dti.to_period('D')
    tm.assert_index_equal(result, expected)