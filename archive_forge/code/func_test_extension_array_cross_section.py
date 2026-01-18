import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
def test_extension_array_cross_section():
    df = DataFrame({'A': pd.array([1, 2], dtype='Int64'), 'B': pd.array([3, 4], dtype='Int64')}, index=['a', 'b'])
    expected = Series(pd.array([1, 3], dtype='Int64'), index=['A', 'B'], name='a')
    result = df.loc['a']
    tm.assert_series_equal(result, expected)
    result = df.iloc[0]
    tm.assert_series_equal(result, expected)