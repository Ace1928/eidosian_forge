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
@pytest.mark.parametrize('klass', [np.array, list])
def test_na_to_datetime(nulls_fixture, klass):
    if isinstance(nulls_fixture, Decimal):
        with pytest.raises(TypeError, match='not convertible to datetime'):
            to_datetime(klass([nulls_fixture]))
    else:
        result = to_datetime(klass([nulls_fixture]))
        assert result[0] is NaT