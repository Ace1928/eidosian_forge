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
def test_dti_constructor_with_non_nano_now_today(self):
    now = Timestamp.now()
    today = Timestamp.today()
    result = DatetimeIndex(['now', 'today'], dtype='M8[s]')
    assert result.dtype == 'M8[s]'
    tolerance = pd.Timedelta(microseconds=1)
    diff0 = result[0] - now.as_unit('s')
    assert diff0 >= pd.Timedelta(0)
    assert diff0 < tolerance
    diff1 = result[1] - today.as_unit('s')
    assert diff1 >= pd.Timedelta(0)
    assert diff1 < tolerance