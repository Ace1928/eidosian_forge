from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz, dtype', [['US/Pacific', 'datetime64[ns, US/Pacific]'], [None, 'datetime64[ns]']])
def test_integer_index_astype_datetime(self, tz, dtype):
    val = [Timestamp('2018-01-01', tz=tz).as_unit('ns')._value]
    result = Index(val, name='idx').astype(dtype)
    expected = DatetimeIndex(['2018-01-01'], tz=tz, name='idx').as_unit('ns')
    tm.assert_index_equal(result, expected)