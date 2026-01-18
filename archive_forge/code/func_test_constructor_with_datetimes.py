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
def test_constructor_with_datetimes(self, using_infer_string):
    intname = np.dtype(int).name
    floatname = np.dtype(np.float64).name
    objectname = np.dtype(np.object_).name
    df = DataFrame({'A': 1, 'B': 'foo', 'C': 'bar', 'D': Timestamp('20010101'), 'E': datetime(2001, 1, 2, 0, 0)}, index=np.arange(10))
    result = df.dtypes
    expected = Series([np.dtype('int64')] + [np.dtype(objectname) if not using_infer_string else 'string'] * 2 + [np.dtype('M8[s]'), np.dtype('M8[us]')], index=list('ABCDE'))
    tm.assert_series_equal(result, expected)
    df = DataFrame({'a': 1.0, 'b': 2, 'c': 'foo', floatname: np.array(1.0, dtype=floatname), intname: np.array(1, dtype=intname)}, index=np.arange(10))
    result = df.dtypes
    expected = Series([np.dtype('float64')] + [np.dtype('int64')] + [np.dtype('object') if not using_infer_string else 'string'] + [np.dtype('float64')] + [np.dtype(intname)], index=['a', 'b', 'c', floatname, intname])
    tm.assert_series_equal(result, expected)
    df = DataFrame({'a': 1.0, 'b': 2, 'c': 'foo', floatname: np.array([1.0] * 10, dtype=floatname), intname: np.array([1] * 10, dtype=intname)}, index=np.arange(10))
    result = df.dtypes
    expected = Series([np.dtype('float64')] + [np.dtype('int64')] + [np.dtype('object') if not using_infer_string else 'string'] + [np.dtype('float64')] + [np.dtype(intname)], index=['a', 'b', 'c', floatname, intname])
    tm.assert_series_equal(result, expected)