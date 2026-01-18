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
def test_loc_getitem_partial_string_slicing_datetimeindex(self):
    df = DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1, 2, 3]}, index=to_datetime(['2020-08-01', '2020-07-02', '2020-08-05']))
    expected = DataFrame({'col1': ['a', 'c'], 'col2': [1, 3]}, index=to_datetime(['2020-08-01', '2020-08-05']))
    result = df.loc['2020-08']
    tm.assert_frame_equal(result, expected)