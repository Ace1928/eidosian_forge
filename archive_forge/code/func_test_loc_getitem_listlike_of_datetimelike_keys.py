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
@pytest.mark.parametrize('to_period', [True, False])
def test_loc_getitem_listlike_of_datetimelike_keys(self, to_period):
    idx = date_range('2011-01-01', '2011-01-02', freq='D', name='idx')
    if to_period:
        idx = idx.to_period('D')
    ser = Series([0.1, 0.2], index=idx, name='s')
    keys = [Timestamp('2011-01-01'), Timestamp('2011-01-02')]
    if to_period:
        keys = [x.to_period('D') for x in keys]
    result = ser.loc[keys]
    exp = Series([0.1, 0.2], index=idx, name='s')
    if not to_period:
        exp.index = exp.index._with_freq(None)
    tm.assert_series_equal(result, exp, check_index_type=True)
    keys = [Timestamp('2011-01-02'), Timestamp('2011-01-02'), Timestamp('2011-01-01')]
    if to_period:
        keys = [x.to_period('D') for x in keys]
    exp = Series([0.2, 0.2, 0.1], index=Index(keys, name='idx', dtype=idx.dtype), name='s')
    result = ser.loc[keys]
    tm.assert_series_equal(result, exp, check_index_type=True)
    keys = [Timestamp('2011-01-03'), Timestamp('2011-01-02'), Timestamp('2011-01-03')]
    if to_period:
        keys = [x.to_period('D') for x in keys]
    with pytest.raises(KeyError, match='not in index'):
        ser.loc[keys]