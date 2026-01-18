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
def test_loc_setitem_empty_frame(self):
    keys1 = ['@' + str(i) for i in range(5)]
    val1 = np.arange(5, dtype='int64')
    keys2 = ['@' + str(i) for i in range(4)]
    val2 = np.arange(4, dtype='int64')
    index = list(set(keys1).union(keys2))
    df = DataFrame(index=index)
    df['A'] = np.nan
    df.loc[keys1, 'A'] = val1
    df['B'] = np.nan
    df.loc[keys2, 'B'] = val2
    sera = Series(val1, index=keys1, dtype=np.float64)
    serb = Series(val2, index=keys2)
    expected = DataFrame({'A': sera, 'B': serb}, columns=Index(['A', 'B'], dtype=object)).reindex(index=index)
    tm.assert_frame_equal(df, expected)