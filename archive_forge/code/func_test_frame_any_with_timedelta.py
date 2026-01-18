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
def test_frame_any_with_timedelta(self):
    df = DataFrame({'a': Series([0, 0]), 't': Series([to_timedelta(0, 's'), to_timedelta(1, 'ms')])})
    result = df.any(axis=0)
    expected = Series(data=[False, True], index=['a', 't'])
    tm.assert_series_equal(result, expected)
    result = df.any(axis=1)
    expected = Series(data=[False, True])
    tm.assert_series_equal(result, expected)