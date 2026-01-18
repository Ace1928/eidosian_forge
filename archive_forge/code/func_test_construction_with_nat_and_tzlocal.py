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
def test_construction_with_nat_and_tzlocal(self):
    tz = dateutil.tz.tzlocal()
    result = DatetimeIndex(['2018', 'NaT'], tz=tz)
    expected = DatetimeIndex([Timestamp('2018', tz=tz), pd.NaT])
    tm.assert_index_equal(result, expected)