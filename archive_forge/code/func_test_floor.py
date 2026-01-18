from datetime import datetime
from dateutil.tz import gettz
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
import pandas.util._test_decorators as td
import pandas._testing as tm
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
def test_floor(self, unit):
    dt = Timestamp('20130101 09:10:11').as_unit(unit)
    result = dt.floor('D')
    expected = Timestamp('20130101')
    assert result == expected
    assert result._creso == dt._creso