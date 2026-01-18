import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_multi_index_left(self, trades, quotes):
    trades = trades.set_index(['time', 'price'])
    quotes = quotes.set_index('time')
    with pytest.raises(MergeError, match='left can only have one index'):
        merge_asof(trades, quotes, left_index=True, right_index=True)