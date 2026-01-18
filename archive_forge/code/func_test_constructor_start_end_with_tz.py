from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('tz', [None, 'America/Los_Angeles', pytz.timezone('America/Los_Angeles'), Timestamp('2000', tz='America/Los_Angeles').tz])
def test_constructor_start_end_with_tz(self, tz):
    start = Timestamp('2013-01-01 06:00:00', tz='America/Los_Angeles')
    end = Timestamp('2013-01-02 06:00:00', tz='America/Los_Angeles')
    result = date_range(freq='D', start=start, end=end, tz=tz)
    expected = DatetimeIndex(['2013-01-01 06:00:00', '2013-01-02 06:00:00'], dtype='M8[ns, America/Los_Angeles]', freq='D')
    tm.assert_index_equal(result, expected)
    assert pytz.timezone('America/Los_Angeles') is result.tz