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
def test_loc_setitem_incremental_with_dst(self):
    base = datetime(2015, 11, 1, tzinfo=gettz('US/Pacific'))
    idxs = [base + timedelta(seconds=i * 900) for i in range(16)]
    result = Series([0], index=[idxs[0]])
    for ts in idxs:
        result.loc[ts] = 1
    expected = Series(1, index=idxs)
    tm.assert_series_equal(result, expected)