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
def test_loc_setitem_multiindex_slice(self):
    index = MultiIndex.from_tuples(zip(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']), names=['first', 'second'])
    result = Series([1, 1, 1, 1, 1, 1, 1, 1], index=index)
    result.loc[('baz', 'one'):('foo', 'two')] = 100
    expected = Series([1, 1, 100, 100, 100, 100, 1, 1], index=index)
    tm.assert_series_equal(result, expected)