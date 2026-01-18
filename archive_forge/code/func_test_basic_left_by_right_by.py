import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_basic_left_by_right_by(self, trades, asof, quotes):
    expected = asof
    result = merge_asof(trades, quotes, on='time', left_by='ticker', right_by='ticker')
    tm.assert_frame_equal(result, expected)