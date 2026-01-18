import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('check', [is_categorical_dtype, is_datetime64tz_dtype, is_period_dtype, is_datetime64_ns_dtype, is_datetime64_dtype, is_interval_dtype, is_datetime64_any_dtype, is_string_dtype, is_bool_dtype])
def test_is_dtype_no_warning(check):
    data = pd.DataFrame({'A': [1, 2]})
    warn = None
    msg = f'{check.__name__} is deprecated'
    if check is is_categorical_dtype or check is is_interval_dtype or check is is_datetime64tz_dtype or (check is is_period_dtype):
        warn = DeprecationWarning
    with tm.assert_produces_warning(warn, match=msg):
        check(data)
    with tm.assert_produces_warning(warn, match=msg):
        check(data['A'])