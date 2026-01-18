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
def test_loc_getitem_setitem_integer_slice_keyerrors(self):
    ser = Series(np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2)))
    cp = ser.copy()
    cp.iloc[4:10] = 0
    assert (cp.iloc[4:10] == 0).all()
    cp = ser.copy()
    cp.iloc[3:11] = 0
    assert (cp.iloc[3:11] == 0).values.all()
    result = ser.iloc[2:6]
    result2 = ser.loc[3:11]
    expected = ser.reindex([4, 6, 8, 10])
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result2, expected)
    s2 = ser.iloc[list(range(5)) + list(range(9, 4, -1))]
    with pytest.raises(KeyError, match='^3$'):
        s2.loc[3:11]
    with pytest.raises(KeyError, match='^3$'):
        s2.loc[3:11] = 0