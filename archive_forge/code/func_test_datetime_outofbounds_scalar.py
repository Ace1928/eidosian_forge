import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
@pytest.mark.parametrize('value', ['3000/12/11 00:00:00'])
@pytest.mark.parametrize('format', [None, '%H:%M:%S'])
def test_datetime_outofbounds_scalar(self, value, format):
    res = to_datetime(value, errors='ignore', format=format)
    assert res == value
    res = to_datetime(value, errors='coerce', format=format)
    assert res is NaT
    if format is not None:
        msg = '^time data ".*" doesn\\\'t match format ".*", at position 0.'
        with pytest.raises(ValueError, match=msg):
            to_datetime(value, errors='raise', format=format)
    else:
        msg = '^Out of bounds .*, at position 0$'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(value, errors='raise', format=format)