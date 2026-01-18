from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_loc_setitem_dt64tz_values(self):
    ser = Series(date_range('2011-01-01', periods=3, tz='US/Eastern'), index=['a', 'b', 'c'])
    s2 = ser.copy()
    expected = Timestamp('2011-01-03', tz='US/Eastern')
    s2.loc['a'] = expected
    result = s2.loc['a']
    assert result == expected
    s2 = ser.copy()
    s2.iloc[0] = expected
    result = s2.iloc[0]
    assert result == expected
    s2 = ser.copy()
    s2['a'] = expected
    result = s2['a']
    assert result == expected