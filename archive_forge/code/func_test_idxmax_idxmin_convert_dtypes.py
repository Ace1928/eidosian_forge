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
@pytest.mark.parametrize('op, expected_value', [('idxmax', [0, 4]), ('idxmin', [0, 5])])
def test_idxmax_idxmin_convert_dtypes(self, op, expected_value):
    df = DataFrame({'ID': [100, 100, 100, 200, 200, 200], 'value': [0, 0, 0, 1, 2, 0]}, dtype='Int64')
    df = df.groupby('ID')
    result = getattr(df, op)()
    expected = DataFrame({'value': expected_value}, index=Index([100, 200], name='ID', dtype='Int64'))
    tm.assert_frame_equal(result, expected)