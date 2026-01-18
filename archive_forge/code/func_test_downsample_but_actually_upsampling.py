from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_downsample_but_actually_upsampling():
    rng = date_range('1/1/2012', periods=100, freq='s')
    ts = Series(np.arange(len(rng), dtype='int64'), index=rng)
    result = ts.resample('20s').asfreq()
    expected = Series([0, 20, 40, 60, 80], index=date_range('2012-01-01 00:00:00', freq='20s', periods=5))
    tm.assert_series_equal(result, expected)