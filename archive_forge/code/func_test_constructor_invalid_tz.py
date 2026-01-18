import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
def test_constructor_invalid_tz(self):
    msg = "Argument 'tzinfo' has incorrect type \\(expected datetime.tzinfo, got str\\)"
    with pytest.raises(TypeError, match=msg):
        Timestamp('2017-10-22', tzinfo='US/Eastern')
    msg = 'at most one of'
    with pytest.raises(ValueError, match=msg):
        Timestamp('2017-10-22', tzinfo=pytz.utc, tz='UTC')
    msg = 'Cannot pass a date attribute keyword argument when passing a date string'
    with pytest.raises(ValueError, match=msg):
        Timestamp('2012-01-01', 'US/Pacific')