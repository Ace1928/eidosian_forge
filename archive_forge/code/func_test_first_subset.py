import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_first_subset(self, frame_or_series):
    ts = DataFrame(np.random.default_rng(2).standard_normal((100, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=100, freq='12h'))
    ts = tm.get_obj(ts, frame_or_series)
    with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
        result = ts.first('10d')
        assert len(result) == 20
    ts = DataFrame(np.random.default_rng(2).standard_normal((100, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=100, freq='D'))
    ts = tm.get_obj(ts, frame_or_series)
    with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
        result = ts.first('10d')
        assert len(result) == 10
    with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
        result = ts.first('3ME')
        expected = ts[:'3/31/2000']
        tm.assert_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
        result = ts.first('21D')
        expected = ts[:21]
        tm.assert_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
        result = ts[:0].first('3ME')
        tm.assert_equal(result, ts[:0])