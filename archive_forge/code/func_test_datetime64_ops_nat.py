from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_datetime64_ops_nat(self, unit):
    datetime_series = Series([NaT, Timestamp('19900315')]).dt.as_unit(unit)
    nat_series_dtype_timestamp = Series([NaT, NaT], dtype=f'datetime64[{unit}]')
    single_nat_dtype_datetime = Series([NaT], dtype=f'datetime64[{unit}]')
    tm.assert_series_equal(-NaT + datetime_series, nat_series_dtype_timestamp)
    msg = "bad operand type for unary -: 'DatetimeArray'"
    with pytest.raises(TypeError, match=msg):
        -single_nat_dtype_datetime + datetime_series
    tm.assert_series_equal(-NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp)
    with pytest.raises(TypeError, match=msg):
        -single_nat_dtype_datetime + nat_series_dtype_timestamp
    tm.assert_series_equal(nat_series_dtype_timestamp + NaT, nat_series_dtype_timestamp)
    tm.assert_series_equal(NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp)
    tm.assert_series_equal(nat_series_dtype_timestamp + NaT, nat_series_dtype_timestamp)
    tm.assert_series_equal(NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp)