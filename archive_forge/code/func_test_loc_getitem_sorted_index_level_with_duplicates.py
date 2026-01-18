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
def test_loc_getitem_sorted_index_level_with_duplicates(self):
    mi = MultiIndex.from_tuples([('foo', 'bar'), ('foo', 'bar'), ('bah', 'bam'), ('bah', 'bam'), ('foo', 'bar'), ('bah', 'bam')], names=['A', 'B'])
    df = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3], [4.0, 4], [5.0, 5], [6.0, 6]], index=mi, columns=['C', 'D'])
    df = df.sort_index(level=0)
    expected = DataFrame([[1.0, 1], [2.0, 2], [5.0, 5]], columns=['C', 'D'], index=mi.take([0, 1, 4]))
    result = df.loc['foo', 'bar']
    tm.assert_frame_equal(result, expected)