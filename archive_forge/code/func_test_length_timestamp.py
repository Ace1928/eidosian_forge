import numpy as np
import pytest
from pandas import (
@pytest.mark.parametrize('left, right, expected', [('2017-01-01', '2017-01-06', '5 days'), ('2017-01-01', '2017-01-01 12:00:00', '12 hours'), ('2017-01-01 12:00', '2017-01-01 12:00:00', '0 days'), ('2017-01-01 12:01', '2017-01-05 17:31:00', '4 days 5 hours 30 min')])
@pytest.mark.parametrize('tz', (None, 'UTC', 'CET', 'US/Eastern'))
def test_length_timestamp(self, tz, left, right, expected):
    iv = Interval(Timestamp(left, tz=tz), Timestamp(right, tz=tz))
    result = iv.length
    expected = Timedelta(expected)
    assert result == expected