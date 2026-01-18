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
def test_loc_setitem_listlike_of_ints(self):
    ser = Series(np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2)))
    inds = [0, 4, 6]
    arr_inds = np.array([0, 4, 6])
    cp = ser.copy()
    exp = ser.copy()
    ser[inds] = 0
    ser.loc[inds] = 0
    tm.assert_series_equal(cp, exp)
    cp = ser.copy()
    exp = ser.copy()
    ser[arr_inds] = 0
    ser.loc[arr_inds] = 0
    tm.assert_series_equal(cp, exp)
    inds_notfound = [0, 4, 5, 6]
    arr_inds_notfound = np.array([0, 4, 5, 6])
    msg = '\\[5\\] not in index'
    with pytest.raises(KeyError, match=msg):
        ser[inds_notfound] = 0
    with pytest.raises(Exception, match=msg):
        ser[arr_inds_notfound] = 0