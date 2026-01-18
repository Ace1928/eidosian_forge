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
@pytest.mark.parametrize('offset,utc,exp', [['Z', True, '2019-01-01T00:00:00.000Z'], ['Z', None, '2019-01-01T00:00:00.000Z'], ['-01:00', True, '2019-01-01T01:00:00.000Z'], ['-01:00', None, '2019-01-01T00:00:00.000-01:00']])
def test_arg_tz_ns_unit(self, offset, utc, exp):
    arg = '2019-01-01T00:00:00.000' + offset
    result = to_datetime([arg], unit='ns', utc=utc)
    expected = to_datetime([exp]).as_unit('ns')
    tm.assert_index_equal(result, expected)