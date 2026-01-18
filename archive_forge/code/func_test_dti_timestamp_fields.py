import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
@pytest.mark.parametrize('field', ['dayofweek', 'day_of_week', 'dayofyear', 'day_of_year', 'quarter', 'days_in_month', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end'])
def test_dti_timestamp_fields(self, field):
    idx = date_range('2020-01-01', periods=10)
    expected = getattr(idx, field)[-1]
    result = getattr(Timestamp(idx[-1]), field)
    assert result == expected