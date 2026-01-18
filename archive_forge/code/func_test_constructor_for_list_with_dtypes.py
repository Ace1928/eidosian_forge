import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def test_constructor_for_list_with_dtypes(self, using_infer_string):
    df = DataFrame([np.arange(5) for x in range(5)])
    result = df.dtypes
    expected = Series([np.dtype('int')] * 5)
    tm.assert_series_equal(result, expected)
    df = DataFrame([np.array(np.arange(5), dtype='int32') for x in range(5)])
    result = df.dtypes
    expected = Series([np.dtype('int32')] * 5)
    tm.assert_series_equal(result, expected)
    df = DataFrame({'a': [2 ** 31, 2 ** 31 + 1]})
    assert df.dtypes.iloc[0] == np.dtype('int64')
    df = DataFrame([1, 2])
    assert df.dtypes.iloc[0] == np.dtype('int64')
    df = DataFrame([1.0, 2.0])
    assert df.dtypes.iloc[0] == np.dtype('float64')
    df = DataFrame({'a': [1, 2]})
    assert df.dtypes.iloc[0] == np.dtype('int64')
    df = DataFrame({'a': [1.0, 2.0]})
    assert df.dtypes.iloc[0] == np.dtype('float64')
    df = DataFrame({'a': 1}, index=range(3))
    assert df.dtypes.iloc[0] == np.dtype('int64')
    df = DataFrame({'a': 1.0}, index=range(3))
    assert df.dtypes.iloc[0] == np.dtype('float64')
    df = DataFrame({'a': [1, 2, 4, 7], 'b': [1.2, 2.3, 5.1, 6.3], 'c': list('abcd'), 'd': [datetime(2000, 1, 1) for i in range(4)], 'e': [1.0, 2, 4.0, 7]})
    result = df.dtypes
    expected = Series([np.dtype('int64'), np.dtype('float64'), np.dtype('object') if not using_infer_string else 'string', np.dtype('datetime64[ns]'), np.dtype('float64')], index=list('abcde'))
    tm.assert_series_equal(result, expected)