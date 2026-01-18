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
@pytest.mark.filterwarnings('ignore:indexing past lexsort depth')
def test_loc_setitem_with_expansion_nonunique_index(self, index):
    if not len(index):
        pytest.skip('Not relevant for empty Index')
    index = index.repeat(2)
    N = len(index)
    arr = np.arange(N).astype(np.int64)
    orig = DataFrame(arr, index=index, columns=[0])
    key = 'kapow'
    assert key not in index
    exp_index = index.insert(len(index), key)
    if isinstance(index, MultiIndex):
        assert exp_index[-1][0] == key
    else:
        assert exp_index[-1] == key
    exp_data = np.arange(N + 1).astype(np.float64)
    expected = DataFrame(exp_data, index=exp_index, columns=[0])
    df = orig.copy()
    df.loc[key, 0] = N
    tm.assert_frame_equal(df, expected)
    ser = orig.copy()[0]
    ser.loc[key] = N
    expected = expected[0].astype(np.int64)
    tm.assert_series_equal(ser, expected)
    df = orig.copy()
    df.loc[key, 1] = N
    expected = DataFrame({0: list(arr) + [np.nan], 1: [np.nan] * N + [float(N)]}, index=exp_index)
    tm.assert_frame_equal(df, expected)