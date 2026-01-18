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
def test_loc_getitem_preserves_index_level_category_dtype(self):
    df = DataFrame(data=np.arange(2, 22, 2), index=MultiIndex(levels=[CategoricalIndex(['a', 'b']), range(10)], codes=[[0] * 5 + [1] * 5, range(10)], names=['Index1', 'Index2']))
    expected = CategoricalIndex(['a', 'b'], categories=['a', 'b'], ordered=False, name='Index1', dtype='category')
    result = df.index.levels[0]
    tm.assert_index_equal(result, expected)
    result = df.loc[['a']].index.levels[0]
    tm.assert_index_equal(result, expected)