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
@pytest.mark.filterwarnings("ignore:'any' with datetime64 dtypes is deprecated.*:FutureWarning")
def test_any_datetime(self):
    float_data = [1, np.nan, 3, np.nan]
    datetime_data = [Timestamp('1960-02-15'), Timestamp('1960-02-16'), pd.NaT, pd.NaT]
    df = DataFrame({'A': float_data, 'B': datetime_data})
    result = df.any(axis=1)
    expected = Series([True, True, True, False])
    tm.assert_series_equal(result, expected)