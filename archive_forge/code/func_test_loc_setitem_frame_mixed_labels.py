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
def test_loc_setitem_frame_mixed_labels(self):
    df = DataFrame({1: [1, 2], 2: [3, 4], 'a': ['a', 'b']})
    result = df.loc[0, [1, 2]]
    expected = Series([1, 3], index=Index([1, 2], dtype=object), dtype=object, name=0)
    tm.assert_series_equal(result, expected)
    expected = DataFrame({1: [5, 2], 2: [6, 4], 'a': ['a', 'b']})
    df.loc[0, [1, 2]] = [5, 6]
    tm.assert_frame_equal(df, expected)