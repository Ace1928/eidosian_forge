from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
def test_date_range_multiplication_overflow(self):
    with tm.assert_produces_warning(None):
        dti = date_range(start='1677-09-22', periods=213503, freq='D')
    assert dti[0] == Timestamp('1677-09-22')
    assert len(dti) == 213503
    msg = 'Cannot generate range with'
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        date_range('1969-05-04', periods=200000000, freq='30000D')