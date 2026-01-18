import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_dt64tz_to_str(self, timezone_frame):
    result = timezone_frame.astype(str)
    expected = DataFrame([['2013-01-01', '2013-01-01 00:00:00-05:00', '2013-01-01 00:00:00+01:00'], ['2013-01-02', 'NaT', 'NaT'], ['2013-01-03', '2013-01-03 00:00:00-05:00', '2013-01-03 00:00:00+01:00']], columns=timezone_frame.columns, dtype='object')
    tm.assert_frame_equal(result, expected)
    with option_context('display.max_columns', 20):
        result = str(timezone_frame)
        assert '0 2013-01-01 2013-01-01 00:00:00-05:00 2013-01-01 00:00:00+01:00' in result
        assert '1 2013-01-02                       NaT                       NaT' in result
        assert '2 2013-01-03 2013-01-03 00:00:00-05:00 2013-01-03 00:00:00+01:00' in result