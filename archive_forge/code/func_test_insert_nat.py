from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('null', [None, np.nan, np.datetime64('NaT'), NaT, NA])
@pytest.mark.parametrize('tz', [None, 'UTC', 'US/Eastern'])
def test_insert_nat(self, tz, null):
    idx = DatetimeIndex(['2017-01-01'], tz=tz)
    expected = DatetimeIndex(['NaT', '2017-01-01'], tz=tz)
    if tz is not None and isinstance(null, np.datetime64):
        expected = Index([null, idx[0]], dtype=object)
    res = idx.insert(0, null)
    tm.assert_index_equal(res, expected)