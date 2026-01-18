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
@pytest.mark.parametrize('unit', ['Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us'])
def test_loc_assign_non_ns_datetime(self, unit):
    df = DataFrame({'timestamp': [np.datetime64('2017-02-11 12:41:29'), np.datetime64('1991-11-07 04:22:37')]})
    df.loc[:, unit] = df.loc[:, 'timestamp'].values.astype(f'datetime64[{unit}]')
    df['expected'] = df.loc[:, 'timestamp'].values.astype(f'datetime64[{unit}]')
    expected = Series(df.loc[:, 'expected'], name=unit)
    tm.assert_series_equal(df.loc[:, unit], expected)