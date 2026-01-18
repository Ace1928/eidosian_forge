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
def test_loc_setitem_range_key(self, frame_or_series):
    obj = frame_or_series(range(5), index=[3, 4, 1, 0, 2])
    values = [9, 10, 11]
    if obj.ndim == 2:
        values = [[9], [10], [11]]
    obj.loc[range(3)] = values
    expected = frame_or_series([0, 1, 10, 9, 11], index=obj.index)
    tm.assert_equal(obj, expected)