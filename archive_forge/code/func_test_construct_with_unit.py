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
@pytest.mark.parametrize('value, check_kwargs', [[946688461000000000, {}], [946688461000000000 / 1000, {'unit': 'us'}], [946688461000000000 / 1000000, {'unit': 'ms'}], [946688461000000000 / 1000000000, {'unit': 's'}], [10957, {'unit': 'D', 'h': 0}], [(946688461000000000 + 500000) / 1000000000, {'unit': 's', 'us': 499, 'ns': 964}], [(946688461000000000 + 500000000) / 1000000000, {'unit': 's', 'us': 500000}], [(946688461000000000 + 500000) / 1000000, {'unit': 'ms', 'us': 500}], [(946688461000000000 + 500000) / 1000, {'unit': 'us', 'us': 500}], [(946688461000000000 + 500000000) / 1000000, {'unit': 'ms', 'us': 500000}], [946688461000000000 / 1000.0 + 5, {'unit': 'us', 'us': 5}], [946688461000000000 / 1000.0 + 5000, {'unit': 'us', 'us': 5000}], [946688461000000000 / 1000000.0 + 0.5, {'unit': 'ms', 'us': 500}], [946688461000000000 / 1000000.0 + 0.005, {'unit': 'ms', 'us': 5, 'ns': 5}], [946688461000000000 / 1000000000.0 + 0.5, {'unit': 's', 'us': 500000}], [10957 + 0.5, {'unit': 'D', 'h': 12}]])
def test_construct_with_unit(self, value, check_kwargs):

    def check(value, unit=None, h=1, s=1, us=0, ns=0):
        stamp = Timestamp(value, unit=unit)
        assert stamp.year == 2000
        assert stamp.month == 1
        assert stamp.day == 1
        assert stamp.hour == h
        if unit != 'D':
            assert stamp.minute == 1
            assert stamp.second == s
            assert stamp.microsecond == us
        else:
            assert stamp.minute == 0
            assert stamp.second == 0
            assert stamp.microsecond == 0
        assert stamp.nanosecond == ns
    check(value, **check_kwargs)