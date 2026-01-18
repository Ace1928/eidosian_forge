import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_maybe_convert_timedelta():
    pi = PeriodIndex(['2000', '2001'], freq='D')
    offset = offsets.Day(2)
    assert pi._maybe_convert_timedelta(offset) == 2
    assert pi._maybe_convert_timedelta(2) == 2
    offset = offsets.BusinessDay()
    msg = 'Input has different freq=B from PeriodIndex\\(freq=D\\)'
    with pytest.raises(ValueError, match=msg):
        pi._maybe_convert_timedelta(offset)