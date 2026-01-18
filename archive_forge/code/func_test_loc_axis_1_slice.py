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
def test_loc_axis_1_slice():
    cols = [(yr, m) for yr in [2014, 2015] for m in [7, 8, 9, 10]]
    df = DataFrame(np.ones((10, 8)), index=tuple('ABCDEFGHIJ'), columns=MultiIndex.from_tuples(cols))
    result = df.loc(axis=1)[(2014, 9):(2015, 8)]
    expected = DataFrame(np.ones((10, 4)), index=tuple('ABCDEFGHIJ'), columns=MultiIndex.from_tuples([(2014, 9), (2014, 10), (2015, 7), (2015, 8)]))
    tm.assert_frame_equal(result, expected)