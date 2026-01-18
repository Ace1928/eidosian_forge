from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_nans_count():
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan
    result = obj.rolling(50, min_periods=30).count()
    tm.assert_almost_equal(result.iloc[-1], np.isfinite(obj[10:-10]).astype(float).sum())