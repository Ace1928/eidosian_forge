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
@pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
def test_loc_setitem_frame_multiples(self, warn_copy_on_write):
    df = DataFrame({'A': ['foo', 'bar', 'baz'], 'B': Series(range(3), dtype=np.int64)})
    rhs = df.loc[1:2]
    rhs.index = df.index[0:2]
    df.loc[0:1] = rhs
    expected = DataFrame({'A': ['bar', 'baz', 'baz'], 'B': Series([1, 2, 2], dtype=np.int64)})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'date': date_range('2000-01-01', '2000-01-5'), 'val': Series(range(5), dtype=np.int64)})
    expected = DataFrame({'date': [Timestamp('20000101'), Timestamp('20000102'), Timestamp('20000101'), Timestamp('20000102'), Timestamp('20000103')], 'val': Series([0, 1, 0, 1, 2], dtype=np.int64)})
    rhs = df.loc[0:2]
    rhs.index = df.index[2:5]
    df.loc[2:4] = rhs
    tm.assert_frame_equal(df, expected)