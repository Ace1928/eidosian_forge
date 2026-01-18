import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat import IS64
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_interval_with_nans(self, frame_or_series, indexer_sl):
    index = IntervalIndex([np.nan, np.nan])
    key = index[:-1]
    obj = frame_or_series(range(2), index=index)
    if frame_or_series is DataFrame and indexer_sl is tm.setitem:
        obj = obj.T
    result = indexer_sl(obj)[key]
    expected = obj
    tm.assert_equal(result, expected)