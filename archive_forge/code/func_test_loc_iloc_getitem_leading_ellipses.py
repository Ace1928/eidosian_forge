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
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_loc_iloc_getitem_leading_ellipses(self, series_with_simple_index, indexer):
    obj = series_with_simple_index
    key = 0 if indexer is tm.iloc or len(obj) == 0 else obj.index[0]
    if indexer is tm.loc and obj.index.inferred_type == 'boolean':
        return
    if indexer is tm.loc and isinstance(obj.index, MultiIndex):
        msg = 'MultiIndex does not support indexing with Ellipsis'
        with pytest.raises(NotImplementedError, match=msg):
            result = indexer(obj)[..., [key]]
    elif len(obj) != 0:
        result = indexer(obj)[..., [key]]
        expected = indexer(obj)[[key]]
        tm.assert_series_equal(result, expected)
    key2 = 0 if indexer is tm.iloc else obj.name
    df = obj.to_frame()
    result = indexer(df)[..., [key2]]
    expected = indexer(df)[:, [key2]]
    tm.assert_frame_equal(result, expected)