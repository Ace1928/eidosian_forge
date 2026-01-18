from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_datetime_count(self):
    df = DataFrame({'a': [1, 2, 3] * 2, 'dates': date_range('now', periods=6, freq='min')})
    result = df.groupby('a').dates.count()
    expected = Series([2, 2, 2], index=Index([1, 2, 3], name='a'), name='dates')
    tm.assert_series_equal(result, expected)