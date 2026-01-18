import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_dt64tz(self, timezone_frame):
    expected = np.array([[Timestamp('2013-01-01 00:00:00'), Timestamp('2013-01-02 00:00:00'), Timestamp('2013-01-03 00:00:00')], [Timestamp('2013-01-01 00:00:00-0500', tz='US/Eastern'), NaT, Timestamp('2013-01-03 00:00:00-0500', tz='US/Eastern')], [Timestamp('2013-01-01 00:00:00+0100', tz='CET'), NaT, Timestamp('2013-01-03 00:00:00+0100', tz='CET')]], dtype=object).T
    expected = DataFrame(expected, index=timezone_frame.index, columns=timezone_frame.columns, dtype=object)
    result = timezone_frame.astype(object)
    tm.assert_frame_equal(result, expected)
    msg = 'Cannot use .astype to convert from timezone-aware dtype to timezone-'
    with pytest.raises(TypeError, match=msg):
        timezone_frame.astype('datetime64[ns]')