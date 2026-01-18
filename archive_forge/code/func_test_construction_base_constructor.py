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
def test_construction_base_constructor(self):
    arr = [Timestamp('2011-01-01'), pd.NaT, Timestamp('2011-01-03')]
    tm.assert_index_equal(Index(arr), DatetimeIndex(arr))
    tm.assert_index_equal(Index(np.array(arr)), DatetimeIndex(np.array(arr)))
    arr = [np.nan, pd.NaT, Timestamp('2011-01-03')]
    tm.assert_index_equal(Index(arr), DatetimeIndex(arr))
    tm.assert_index_equal(Index(np.array(arr)), DatetimeIndex(np.array(arr)))