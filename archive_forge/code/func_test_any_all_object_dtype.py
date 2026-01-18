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
@pytest.mark.parametrize('bool_agg_func', ['any', 'all'])
@pytest.mark.parametrize('skipna', [True, False])
def test_any_all_object_dtype(self, axis, bool_agg_func, skipna, using_infer_string):
    df = DataFrame(data=[[1, np.nan, np.nan, True], [np.nan, 2, np.nan, True], [np.nan, np.nan, np.nan, True], [np.nan, np.nan, '5', np.nan]])
    if using_infer_string:
        val = not axis == 0 and (not skipna) and (bool_agg_func == 'all')
    else:
        val = True
    result = getattr(df, bool_agg_func)(axis=axis, skipna=skipna)
    expected = Series([True, True, val, True])
    tm.assert_series_equal(result, expected)