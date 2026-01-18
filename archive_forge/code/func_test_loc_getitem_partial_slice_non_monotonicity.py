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
@pytest.mark.parametrize('indexer_end', [None, '2020-01-02 23:59:59.999999999'])
def test_loc_getitem_partial_slice_non_monotonicity(self, tz_aware_fixture, indexer_end, frame_or_series):
    obj = frame_or_series([1] * 5, index=DatetimeIndex([Timestamp('2019-12-30'), Timestamp('2020-01-01'), Timestamp('2019-12-25'), Timestamp('2020-01-02 23:59:59.999999999'), Timestamp('2019-12-19')], tz=tz_aware_fixture))
    expected = frame_or_series([1] * 2, index=DatetimeIndex([Timestamp('2020-01-01'), Timestamp('2020-01-02 23:59:59.999999999')], tz=tz_aware_fixture))
    indexer = slice('2020-01-01', indexer_end)
    result = obj[indexer]
    tm.assert_equal(result, expected)
    result = obj.loc[indexer]
    tm.assert_equal(result, expected)