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
def test_constructor_dtype_tz_mismatch_raises(self):
    idx = DatetimeIndex(['2013-01-01', '2013-01-02'], dtype='datetime64[ns, US/Eastern]')
    msg = 'cannot supply both a tz and a timezone-naive dtype \\(i\\.e\\. datetime64\\[ns\\]\\)'
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex(idx, dtype='datetime64[ns]')
    msg = 'data is already tz-aware US/Eastern, unable to set specified tz: CET'
    with pytest.raises(TypeError, match=msg):
        DatetimeIndex(idx, dtype='datetime64[ns, CET]')
    msg = 'cannot supply both a tz and a dtype with a tz'
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex(idx, tz='CET', dtype='datetime64[ns, US/Eastern]')
    result = DatetimeIndex(idx, dtype='datetime64[ns, US/Eastern]')
    tm.assert_index_equal(idx, result)