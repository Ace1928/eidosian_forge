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
def test_loc_drops_level(self):
    mi = MultiIndex.from_product([list('ab'), list('xy'), [1, 2]], names=['ab', 'xy', 'num'])
    ser = Series(range(8), index=mi)
    loc_result = ser.loc['a', :, :]
    expected = ser.index.droplevel(0)[:4]
    tm.assert_index_equal(loc_result.index, expected)