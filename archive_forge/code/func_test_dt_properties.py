from __future__ import annotations
from datetime import (
from decimal import Decimal
from io import (
import operator
import pickle
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
from pandas.tests.extension import base
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
@pytest.mark.parametrize('prop, expected', [['year', 2023], ['day', 2], ['day_of_week', 0], ['dayofweek', 0], ['weekday', 0], ['day_of_year', 2], ['dayofyear', 2], ['hour', 3], ['minute', 4], ['is_leap_year', False], ['microsecond', 5], ['month', 1], ['nanosecond', 6], ['quarter', 1], ['second', 7], ['date', date(2023, 1, 2)], ['time', time(3, 4, 7, 5)]])
def test_dt_properties(prop, expected):
    ser = pd.Series([pd.Timestamp(year=2023, month=1, day=2, hour=3, minute=4, second=7, microsecond=5, nanosecond=6), None], dtype=ArrowDtype(pa.timestamp('ns')))
    result = getattr(ser.dt, prop)
    exp_type = None
    if isinstance(expected, date):
        exp_type = pa.date32()
    elif isinstance(expected, time):
        exp_type = pa.time64('ns')
    expected = pd.Series(ArrowExtensionArray(pa.array([expected, None], type=exp_type)))
    tm.assert_series_equal(result, expected)