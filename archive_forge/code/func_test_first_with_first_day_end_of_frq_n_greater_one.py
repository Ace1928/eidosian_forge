import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_first_with_first_day_end_of_frq_n_greater_one(self, frame_or_series):
    x = frame_or_series([1] * 100, index=bdate_range('2010-03-31', periods=100))
    with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
        result = x.first('2ME')
    expected = frame_or_series([1] * 23, index=bdate_range('2010-03-31', '2010-04-30'))
    tm.assert_equal(result, expected)