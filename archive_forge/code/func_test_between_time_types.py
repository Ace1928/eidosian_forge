from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_between_time_types(self, frame_or_series):
    rng = date_range('1/1/2000', '1/5/2000', freq='5min')
    obj = DataFrame({'A': 0}, index=rng)
    obj = tm.get_obj(obj, frame_or_series)
    msg = 'Cannot convert arg \\[datetime\\.datetime\\(2010, 1, 2, 1, 0\\)\\] to a time'
    with pytest.raises(ValueError, match=msg):
        obj.between_time(datetime(2010, 1, 2, 1), datetime(2010, 1, 2, 5))