from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_difference_freq(self, sort):
    index = date_range('20160920', '20160925', freq='D')
    other = date_range('20160921', '20160924', freq='D')
    expected = DatetimeIndex(['20160920', '20160925'], dtype='M8[ns]', freq=None)
    idx_diff = index.difference(other, sort)
    tm.assert_index_equal(idx_diff, expected)
    tm.assert_attr_equal('freq', idx_diff, expected)
    other = date_range('20160922', '20160925', freq='D')
    idx_diff = index.difference(other, sort)
    expected = DatetimeIndex(['20160920', '20160921'], dtype='M8[ns]', freq='D')
    tm.assert_index_equal(idx_diff, expected)
    tm.assert_attr_equal('freq', idx_diff, expected)