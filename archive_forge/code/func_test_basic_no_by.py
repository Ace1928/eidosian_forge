import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_basic_no_by(self, trades, asof, quotes):
    f = lambda x: x[x.ticker == 'MSFT'].drop('ticker', axis=1).reset_index(drop=True)
    expected = f(asof)
    trades = f(trades)
    quotes = f(quotes)
    result = merge_asof(trades, quotes, on='time')
    tm.assert_frame_equal(result, expected)