import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_td64_to_string(self, frame_or_series):
    tdi = pd.timedelta_range('1 Day', periods=3)
    obj = frame_or_series(tdi)
    expected = frame_or_series(['1 days', '2 days', '3 days'], dtype='string')
    result = obj.astype('string')
    tm.assert_equal(result, expected)