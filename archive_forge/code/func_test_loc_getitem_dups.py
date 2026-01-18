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
def test_loc_getitem_dups(self):
    df = DataFrame(np.random.default_rng(2).random((20, 5)), index=['ABCDE'[x % 5] for x in range(20)])
    expected = df.loc['A', 0]
    result = df.loc[:, 0].loc['A']
    tm.assert_series_equal(result, expected)