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
def test_loc_setitem_categorical_values_partial_column_slice(self):
    df = DataFrame({'a': [1, 1, 1, 1, 1], 'b': list('aaaaa')})
    exp = DataFrame({'a': [1, 'b', 'b', 1, 1], 'b': list('aabba')})
    with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
        df.loc[1:2, 'a'] = Categorical(['b', 'b'], categories=['a', 'b'])
        df.loc[2:3, 'b'] = Categorical(['b', 'b'], categories=['a', 'b'])
    tm.assert_frame_equal(df, exp)