from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_date_range_localize2(self, unit):
    rng = date_range('3/11/2012 00:00', periods=2, freq='h', tz='US/Eastern', unit=unit)
    rng2 = DatetimeIndex(['3/11/2012 00:00', '3/11/2012 01:00'], dtype=f'M8[{unit}, US/Eastern]', freq='h')
    tm.assert_index_equal(rng, rng2)
    exp = Timestamp('3/11/2012 00:00', tz='US/Eastern')
    assert exp.hour == 0
    assert rng[0] == exp
    exp = Timestamp('3/11/2012 01:00', tz='US/Eastern')
    assert exp.hour == 1
    assert rng[1] == exp
    rng = date_range('3/11/2012 00:00', periods=10, freq='h', tz='US/Eastern', unit=unit)
    assert rng[2].hour == 3