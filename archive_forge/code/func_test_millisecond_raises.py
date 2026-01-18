import calendar
from datetime import (
import locale
import time
import unicodedata
from dateutil.tz import (
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.timezones import (
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
@pytest.mark.parametrize('tz', [None, 'US/Eastern'])
def test_millisecond_raises(self, tz):
    ts = Timestamp('2014-12-31 23:59:00', tz=tz)
    msg = "'Timestamp' object has no attribute 'millisecond'"
    with pytest.raises(AttributeError, match=msg):
        ts.millisecond