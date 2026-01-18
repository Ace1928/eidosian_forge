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
@td.skip_if_not_us_locale
def test_guess_datetime_format_for_array_all_nans(self):
    format_for_string_of_nans = tools._guess_datetime_format_for_array(np.array([np.nan, np.nan, np.nan], dtype='O'))
    assert format_for_string_of_nans is None