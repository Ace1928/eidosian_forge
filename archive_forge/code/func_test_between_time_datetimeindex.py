from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_between_time_datetimeindex(self):
    index = date_range('2012-01-01', '2012-01-05', freq='30min')
    df = DataFrame(np.random.default_rng(2).standard_normal((len(index), 5)), index=index)
    bkey = slice(time(13, 0, 0), time(14, 0, 0))
    binds = [26, 27, 28, 74, 75, 76, 122, 123, 124, 170, 171, 172]
    result = df.between_time(bkey.start, bkey.stop)
    expected = df.loc[bkey]
    expected2 = df.iloc[binds]
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result, expected2)
    assert len(result) == 12