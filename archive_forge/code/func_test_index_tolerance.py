import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_index_tolerance(self, trades, quotes, tolerance):
    expected = tolerance.set_index('time')
    trades = trades.set_index('time')
    quotes = quotes.set_index('time')
    result = merge_asof(trades, quotes, left_index=True, right_index=True, by='ticker', tolerance=Timedelta('1day'))
    tm.assert_frame_equal(result, expected)