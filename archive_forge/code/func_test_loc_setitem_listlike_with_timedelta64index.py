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
@pytest.mark.parametrize('indexer, expected', [(0, [20, 1, 2, 3, 4, 5, 6, 7, 8, 9]), (slice(4, 8), [0, 1, 2, 3, 20, 20, 20, 20, 8, 9]), ([3, 5], [0, 1, 2, 20, 4, 20, 6, 7, 8, 9])])
def test_loc_setitem_listlike_with_timedelta64index(self, indexer, expected):
    tdi = to_timedelta(range(10), unit='s')
    df = DataFrame({'x': range(10)}, dtype='int64', index=tdi)
    df.loc[df.index[indexer], 'x'] = 20
    expected = DataFrame(expected, index=tdi, columns=['x'], dtype='int64')
    tm.assert_frame_equal(expected, df)