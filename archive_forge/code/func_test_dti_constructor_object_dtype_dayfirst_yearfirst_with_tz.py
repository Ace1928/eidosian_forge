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
def test_dti_constructor_object_dtype_dayfirst_yearfirst_with_tz(self):
    val = '5/10/16'
    dfirst = Timestamp(2016, 10, 5, tz='US/Pacific')
    yfirst = Timestamp(2005, 10, 16, tz='US/Pacific')
    result1 = DatetimeIndex([val], tz='US/Pacific', dayfirst=True)
    expected1 = DatetimeIndex([dfirst])
    tm.assert_index_equal(result1, expected1)
    result2 = DatetimeIndex([val], tz='US/Pacific', yearfirst=True)
    expected2 = DatetimeIndex([yfirst])
    tm.assert_index_equal(result2, expected2)