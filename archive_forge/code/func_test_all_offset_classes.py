from datetime import timedelta
import pytest
import pytz
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas.errors import PerformanceWarning
from pandas import DatetimeIndex
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('tup', offset_classes)
def test_all_offset_classes(self, tup):
    offset, test_values = tup
    first = Timestamp(test_values[0], tz='US/Eastern') + offset()
    second = Timestamp(test_values[1], tz='US/Eastern')
    assert first == second