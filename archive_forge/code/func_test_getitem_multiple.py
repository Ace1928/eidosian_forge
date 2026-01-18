import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_getitem_multiple(self, roll_frame):
    g = roll_frame.groupby('A')
    r = g.rolling(2, min_periods=0)
    g_mutated = get_groupby(roll_frame, by='A')
    expected = g_mutated.B.apply(lambda x: x.rolling(2, min_periods=0).count())
    result = r.B.count()
    tm.assert_series_equal(result, expected)
    result = r.B.count()
    tm.assert_series_equal(result, expected)