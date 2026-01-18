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
def test_loc_getitem_multiindex_tuple_level():
    lev1 = ['a', 'b', 'c']
    lev2 = [(0, 1), (1, 0)]
    lev3 = [0, 1]
    cols = MultiIndex.from_product([lev1, lev2, lev3], names=['x', 'y', 'z'])
    df = DataFrame(6, index=range(5), columns=cols)
    result = df.loc[:, (lev1[0], lev2[0], lev3[0])]
    expected = df.iloc[:, :1]
    tm.assert_frame_equal(result, expected)
    alt = df.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=1)
    tm.assert_frame_equal(alt, expected)
    ser = df.iloc[0]
    expected2 = ser.iloc[:1]
    alt2 = ser.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=0)
    tm.assert_series_equal(alt2, expected2)
    result2 = ser.loc[lev1[0], lev2[0], lev3[0]]
    assert result2 == 6