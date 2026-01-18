import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_allow_exact_matches(self, trades, quotes, allow_exact_matches):
    result = merge_asof(trades, quotes, on='time', by='ticker', allow_exact_matches=False)
    expected = allow_exact_matches
    tm.assert_frame_equal(result, expected)