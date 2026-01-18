from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_between_time_raises(self, frame_or_series):
    obj = DataFrame([[1, 2, 3], [4, 5, 6]])
    obj = tm.get_obj(obj, frame_or_series)
    msg = 'Index must be DatetimeIndex'
    with pytest.raises(TypeError, match=msg):
        obj.between_time(start_time='00:00', end_time='12:00')