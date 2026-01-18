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
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_idxmax_empty(self, index, skipna, axis):
    if axis == 0:
        frame = DataFrame(index=index)
    else:
        frame = DataFrame(columns=index)
    result = frame.idxmax(axis=axis, skipna=skipna)
    expected = Series(dtype=index.dtype)
    tm.assert_series_equal(result, expected)