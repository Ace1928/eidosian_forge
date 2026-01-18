import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_basic_right_index(self, trades, asof, quotes):
    expected = asof
    quotes = quotes.set_index('time')
    result = merge_asof(trades, quotes, left_on='time', right_index=True, by='ticker')
    tm.assert_frame_equal(result, expected)