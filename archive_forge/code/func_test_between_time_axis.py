from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_between_time_axis(self, frame_or_series):
    rng = date_range('1/1/2000', periods=100, freq='10min')
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    if frame_or_series is DataFrame:
        ts = ts.to_frame()
    stime, etime = ('08:00:00', '09:00:00')
    expected_length = 7
    assert len(ts.between_time(stime, etime)) == expected_length
    assert len(ts.between_time(stime, etime, axis=0)) == expected_length
    msg = f'No axis named {ts.ndim} for object type {type(ts).__name__}'
    with pytest.raises(ValueError, match=msg):
        ts.between_time(stime, etime, axis=ts.ndim)