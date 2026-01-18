import dateutil.tz
import numpy as np
import pytest
from pandas import (
from pandas.core.arrays import datetimes
@pytest.mark.parametrize('tz', [None, 'UTC', 'US/Central', dateutil.tz.tzoffset(None, -28800)])
def test_iteration_preserves_nanoseconds(self, tz):
    index = DatetimeIndex(['2018-02-08 15:00:00.168456358', '2018-02-08 15:00:00.168456359'], tz=tz)
    for i, ts in enumerate(index):
        assert ts == index[i]