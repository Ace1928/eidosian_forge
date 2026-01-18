from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
@pytest.mark.parametrize('tz', [None, 'UTC'])
def test_mean_includes_datetimes(self, tz):
    df = DataFrame({'A': [Timestamp('2000', tz=tz)] * 2})
    result = df.mean()
    expected = Series([Timestamp('2000', tz=tz)], index=['A'])
    tm.assert_series_equal(result, expected)