from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_grouper_multilevel_freq(self):
    d0 = date.today() - timedelta(days=14)
    dates = date_range(d0, date.today())
    date_index = MultiIndex.from_product([dates, dates], names=['foo', 'bar'])
    df = DataFrame(np.random.default_rng(2).integers(0, 100, 225), index=date_index)
    expected = df.reset_index().groupby([Grouper(key='foo', freq='W'), Grouper(key='bar', freq='W')]).sum()
    expected.columns = Index([0], dtype='int64')
    result = df.groupby([Grouper(level='foo', freq='W'), Grouper(level='bar', freq='W')]).sum()
    tm.assert_frame_equal(result, expected)
    result = df.groupby([Grouper(level=0, freq='W'), Grouper(level=1, freq='W')]).sum()
    tm.assert_frame_equal(result, expected)