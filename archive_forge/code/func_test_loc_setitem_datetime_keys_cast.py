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
@pytest.mark.parametrize('conv', [lambda x: x, lambda x: x.to_datetime64(), lambda x: x.to_pydatetime(), lambda x: np.datetime64(x)], ids=['self', 'to_datetime64', 'to_pydatetime', 'np.datetime64'])
def test_loc_setitem_datetime_keys_cast(self, conv):
    dt1 = Timestamp('20130101 09:00:00')
    dt2 = Timestamp('20130101 10:00:00')
    df = DataFrame()
    df.loc[conv(dt1), 'one'] = 100
    df.loc[conv(dt2), 'one'] = 200
    expected = DataFrame({'one': [100.0, 200.0]}, index=[dt1, dt2], columns=Index(['one'], dtype=object))
    tm.assert_frame_equal(df, expected)