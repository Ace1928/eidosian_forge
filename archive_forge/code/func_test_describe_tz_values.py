import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_describe_tz_values(self, tz_naive_fixture):
    tz = tz_naive_fixture
    s1 = Series(range(5))
    start = Timestamp(2018, 1, 1)
    end = Timestamp(2018, 1, 5)
    s2 = Series(date_range(start, end, tz=tz))
    df = DataFrame({'s1': s1, 's2': s2})
    expected = DataFrame({'s1': [5, 2, 0, 1, 2, 3, 4, 1.581139], 's2': [5, Timestamp(2018, 1, 3).tz_localize(tz), start.tz_localize(tz), s2[1], s2[2], s2[3], end.tz_localize(tz), np.nan]}, index=['count', 'mean', 'min', '25%', '50%', '75%', 'max', 'std'])
    result = df.describe(include='all')
    tm.assert_frame_equal(result, expected)