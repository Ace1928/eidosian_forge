from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_grouper_creation_bug3(self, unit):
    dti = date_range('20130101', periods=2, unit=unit)
    mi = MultiIndex.from_product([list('ab'), range(2), dti], names=['one', 'two', 'three'])
    ser = Series(np.arange(8, dtype='int64'), index=mi)
    result = ser.groupby(Grouper(level='three', freq='ME')).sum()
    exp_dti = pd.DatetimeIndex([Timestamp('2013-01-31')], freq='ME', name='three').as_unit(unit)
    expected = Series([28], index=exp_dti)
    tm.assert_series_equal(result, expected)
    result = ser.groupby(Grouper(level='one')).sum()
    expected = ser.groupby(level='one').sum()
    tm.assert_series_equal(result, expected)