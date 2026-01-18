from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_between_time_axis_aliases(self, axis):
    rng = date_range('1/1/2000', periods=100, freq='10min')
    ts = DataFrame(np.random.default_rng(2).standard_normal((len(rng), len(rng))))
    stime, etime = ('08:00:00', '09:00:00')
    exp_len = 7
    if axis in ['index', 0]:
        ts.index = rng
        assert len(ts.between_time(stime, etime)) == exp_len
        assert len(ts.between_time(stime, etime, axis=0)) == exp_len
    if axis in ['columns', 1]:
        ts.columns = rng
        selected = ts.between_time(stime, etime, axis=1).columns
        assert len(selected) == exp_len