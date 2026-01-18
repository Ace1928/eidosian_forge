import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import (
from pandas.core.indexers.objects import (
from pandas.tseries.offsets import BusinessDay
def test_fixed_forward_indexer_count(step):
    df = DataFrame({'b': [None, None, None, 7]})
    indexer = FixedForwardWindowIndexer(window_size=2)
    result = df.rolling(window=indexer, min_periods=0, step=step).count()
    expected = DataFrame({'b': [0.0, 0.0, 1.0, 1.0]})[::step]
    tm.assert_frame_equal(result, expected)