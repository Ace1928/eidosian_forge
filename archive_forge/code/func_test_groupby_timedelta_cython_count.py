from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_groupby_timedelta_cython_count():
    df = DataFrame({'g': list('ab' * 2), 'delta': np.arange(4).astype('timedelta64[ns]')})
    expected = Series([2, 2], index=Index(['a', 'b'], name='g'), name='delta')
    result = df.groupby('g').delta.count()
    tm.assert_series_equal(expected, result)