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
def test_construction_from_replaced_timestamps_with_dst(self):
    index = date_range(Timestamp(2000, 12, 31), Timestamp(2005, 12, 31), freq='YE-DEC', tz='Australia/Melbourne')
    result = DatetimeIndex([x.replace(month=6, day=1) for x in index])
    expected = DatetimeIndex(['2000-06-01 00:00:00', '2001-06-01 00:00:00', '2002-06-01 00:00:00', '2003-06-01 00:00:00', '2004-06-01 00:00:00', '2005-06-01 00:00:00'], tz='Australia/Melbourne')
    tm.assert_index_equal(result, expected)