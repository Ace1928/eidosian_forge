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
def test_loc_getitem_multiindex_nonunique_len_zero(self):
    mi = MultiIndex.from_product([[0], [1, 1]])
    ser = Series(0, index=mi)
    res = ser.loc[[]]
    expected = ser[:0]
    tm.assert_series_equal(res, expected)
    res2 = ser.loc[ser.iloc[0:0]]
    tm.assert_series_equal(res2, expected)