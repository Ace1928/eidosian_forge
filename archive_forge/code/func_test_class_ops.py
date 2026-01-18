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
def test_class_ops(self):

    def compare(x, y):
        assert int((Timestamp(x)._value - Timestamp(y)._value) / 1000000000.0) == 0
    compare(Timestamp.now(), datetime.now())
    compare(Timestamp.now('UTC'), datetime.now(pytz.timezone('UTC')))
    compare(Timestamp.now('UTC'), datetime.now(tzutc()))
    compare(Timestamp.utcnow(), datetime.now(timezone.utc))
    compare(Timestamp.today(), datetime.today())
    current_time = calendar.timegm(datetime.now().utctimetuple())
    ts_utc = Timestamp.utcfromtimestamp(current_time)
    assert ts_utc.timestamp() == current_time
    compare(Timestamp.fromtimestamp(current_time), datetime.fromtimestamp(current_time))
    compare(Timestamp.fromtimestamp(current_time, 'UTC'), datetime.fromtimestamp(current_time, utc))
    compare(Timestamp.fromtimestamp(current_time, tz='UTC'), datetime.fromtimestamp(current_time, utc))
    date_component = datetime.now(timezone.utc)
    time_component = (date_component + timedelta(minutes=10)).time()
    compare(Timestamp.combine(date_component, time_component), datetime.combine(date_component, time_component))