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
def test_loc_getitem_label_slice_across_dst(self):
    idx = date_range('2017-10-29 01:30:00', tz='Europe/Berlin', periods=5, freq='30 min')
    series2 = Series([0, 1, 2, 3, 4], index=idx)
    t_1 = Timestamp('2017-10-29 02:30:00+02:00', tz='Europe/Berlin')
    t_2 = Timestamp('2017-10-29 02:00:00+01:00', tz='Europe/Berlin')
    result = series2.loc[t_1:t_2]
    expected = Series([2, 3], index=idx[2:4])
    tm.assert_series_equal(result, expected)
    result = series2[t_1]
    expected = 2
    assert result == expected