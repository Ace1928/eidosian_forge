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
@pytest.mark.parametrize('tz', [pytz.timezone('US/Eastern'), gettz('US/Eastern')])
def test_dti_from_tzaware_datetime(self, tz):
    d = [datetime(2012, 8, 19, tzinfo=tz)]
    index = DatetimeIndex(d)
    assert timezones.tz_compare(index.tz, tz)