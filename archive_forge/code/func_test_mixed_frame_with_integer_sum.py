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
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="sum doesn't work with arrow strings")
def test_mixed_frame_with_integer_sum():
    df = DataFrame([['a', 1]], columns=list('ab'))
    df = df.astype({'b': 'Int64'})
    result = df.sum()
    expected = Series(['a', 1], index=['a', 'b'])
    tm.assert_series_equal(result, expected)