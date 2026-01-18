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
def test_construction_outofbounds(self):
    dates = [datetime(3000, 1, 1), datetime(4000, 1, 1), datetime(5000, 1, 1), datetime(6000, 1, 1)]
    exp = Index(dates, dtype=object)
    tm.assert_index_equal(Index(dates), exp)
    msg = '^Out of bounds nanosecond timestamp: 3000-01-01 00:00:00, at position 0$'
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        DatetimeIndex(dates)