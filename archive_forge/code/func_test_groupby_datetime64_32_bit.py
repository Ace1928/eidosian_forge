from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_groupby_datetime64_32_bit(self):
    df = DataFrame({'A': range(2), 'B': [Timestamp('2000-01-1')] * 2})
    result = df.groupby('A')['B'].transform('min')
    expected = Series([Timestamp('2000-01-1')] * 2, name='B')
    tm.assert_series_equal(result, expected)