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
@pytest.mark.parametrize('start,stop, expected_slice', [[np.timedelta64(0, 'ns'), None, slice(0, 11)], [np.timedelta64(1, 'D'), np.timedelta64(6, 'D'), slice(1, 7)], [None, np.timedelta64(4, 'D'), slice(0, 5)]])
def test_loc_getitem_slice_label_td64obj(self, start, stop, expected_slice):
    ser = Series(range(11), timedelta_range('0 days', '10 days'))
    result = ser.loc[slice(start, stop)]
    expected = ser.iloc[expected_slice]
    tm.assert_series_equal(result, expected)