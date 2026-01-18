import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_describe_tz_values2(self):
    tz = 'CET'
    s1 = Series(range(5))
    start = Timestamp(2018, 1, 1)
    end = Timestamp(2018, 1, 5)
    s2 = Series(date_range(start, end, tz=tz))
    df = DataFrame({'s1': s1, 's2': s2})
    s1_ = s1.describe()
    s2_ = s2.describe()
    idx = ['count', 'mean', 'min', '25%', '50%', '75%', 'max', 'std']
    expected = pd.concat([s1_, s2_], axis=1, keys=['s1', 's2']).reindex(idx, copy=False)
    result = df.describe(include='all')
    tm.assert_frame_equal(result, expected)