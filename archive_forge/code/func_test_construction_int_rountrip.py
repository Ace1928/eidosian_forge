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
def test_construction_int_rountrip(self, tz_naive_fixture):
    tz = tz_naive_fixture
    result = 1293858000000000000
    expected = DatetimeIndex([result], tz=tz).asi8[0]
    assert result == expected