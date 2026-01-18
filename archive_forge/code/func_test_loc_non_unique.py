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
def test_loc_non_unique(self):
    df = DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [3, 4, 5, 6, 7, 8]}, index=[0, 1, 0, 1, 2, 3])
    msg = "'Cannot get left slice bound for non-unique label: 1'"
    with pytest.raises(KeyError, match=msg):
        df.loc[1:]
    msg = "'Cannot get left slice bound for non-unique label: 0'"
    with pytest.raises(KeyError, match=msg):
        df.loc[0:]
    msg = "'Cannot get left slice bound for non-unique label: 1'"
    with pytest.raises(KeyError, match=msg):
        df.loc[1:2]
    df = DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [3, 4, 5, 6, 7, 8]}, index=[0, 1, 0, 1, 2, 3]).sort_index(axis=0)
    result = df.loc[1:]
    expected = DataFrame({'A': [2, 4, 5, 6], 'B': [4, 6, 7, 8]}, index=[1, 1, 2, 3])
    tm.assert_frame_equal(result, expected)
    result = df.loc[0:]
    tm.assert_frame_equal(result, df)
    result = df.loc[1:2]
    expected = DataFrame({'A': [2, 4, 5], 'B': [4, 6, 7]}, index=[1, 1, 2])
    tm.assert_frame_equal(result, expected)