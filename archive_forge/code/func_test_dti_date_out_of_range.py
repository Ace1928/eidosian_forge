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
@pytest.mark.parametrize('data', [['1400-01-01'], [datetime(1400, 1, 1)]])
def test_dti_date_out_of_range(self, data):
    msg = '^Out of bounds nanosecond timestamp: 1400-01-01( 00:00:00)?, at position 0$'
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        DatetimeIndex(data)