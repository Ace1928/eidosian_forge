from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_constructor_datetimes_mixed_tzs(self):
    tz = maybe_get_tz('US/Central')
    dt1 = datetime(2020, 1, 1, tzinfo=tz)
    dt2 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    result = Index([dt1, dt2])
    expected = Index([dt1, dt2], dtype=object)
    tm.assert_index_equal(result, expected)