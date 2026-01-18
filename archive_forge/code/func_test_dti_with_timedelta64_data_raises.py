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
def test_dti_with_timedelta64_data_raises(self):
    data = np.array([0], dtype='m8[ns]')
    msg = 'timedelta64\\[ns\\] cannot be converted to datetime64'
    with pytest.raises(TypeError, match=msg):
        DatetimeIndex(data)
    with pytest.raises(TypeError, match=msg):
        to_datetime(data)
    with pytest.raises(TypeError, match=msg):
        DatetimeIndex(pd.TimedeltaIndex(data))
    with pytest.raises(TypeError, match=msg):
        to_datetime(pd.TimedeltaIndex(data))