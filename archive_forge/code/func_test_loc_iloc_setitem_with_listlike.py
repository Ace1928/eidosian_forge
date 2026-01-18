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
@pytest.mark.parametrize('array_fn', [np.array, pd.array, list, tuple])
@pytest.mark.parametrize('size', [0, 4, 5, 6])
def test_loc_iloc_setitem_with_listlike(self, size, array_fn):
    arr = array_fn([0] * size)
    expected = Series([arr, 0, 0, 0, 0], index=list('abcde'), dtype=object)
    ser = Series(0, index=list('abcde'), dtype=object)
    ser.loc['a'] = arr
    tm.assert_series_equal(ser, expected)
    ser = Series(0, index=list('abcde'), dtype=object)
    ser.iloc[0] = arr
    tm.assert_series_equal(ser, expected)