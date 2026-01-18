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
def test_min_max_dt64_api_consistency_empty_df(self):
    df = DataFrame({'x': []})
    expected_float_series = Series([], dtype=float)
    assert np.isnan(df.min(axis=0).x) == np.isnan(expected_float_series.min())
    assert np.isnan(df.max(axis=0).x) == np.isnan(expected_float_series.max())
    tm.assert_series_equal(df.min(axis=1), expected_float_series)
    tm.assert_series_equal(df.min(axis=1), expected_float_series)