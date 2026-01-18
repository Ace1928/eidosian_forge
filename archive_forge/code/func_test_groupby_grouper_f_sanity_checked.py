from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_grouper_f_sanity_checked(self):
    dates = date_range('01-Jan-2013', periods=12, freq='MS')
    ts = Series(np.random.default_rng(2).standard_normal(12), index=dates)
    msg = "'Timestamp' object is not subscriptable"
    with pytest.raises(TypeError, match=msg):
        ts.groupby(lambda key: key[0:6])
    result = ts.groupby(lambda x: x).sum()
    expected = ts.groupby(ts.index).sum()
    expected.index.freq = None
    tm.assert_series_equal(result, expected)