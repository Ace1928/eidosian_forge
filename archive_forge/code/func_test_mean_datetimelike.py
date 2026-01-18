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
def test_mean_datetimelike(self):
    df = DataFrame({'A': np.arange(3), 'B': date_range('2016-01-01', periods=3), 'C': pd.timedelta_range('1D', periods=3), 'D': pd.period_range('2016', periods=3, freq='Y')})
    result = df.mean(numeric_only=True)
    expected = Series({'A': 1.0})
    tm.assert_series_equal(result, expected)
    with pytest.raises(TypeError, match='mean is not implemented for PeriodArray'):
        df.mean()