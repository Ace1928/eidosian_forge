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
def test_constructor_categorical_series(self):
    items = [1, 2, 3, 1]
    exp = Series(items).astype('category')
    res = Series(items, dtype='category')
    tm.assert_series_equal(res, exp)
    items = ['a', 'b', 'c', 'a']
    exp = Series(items).astype('category')
    res = Series(items, dtype='category')
    tm.assert_series_equal(res, exp)
    index = date_range('20000101', periods=3)
    expected = Series(Categorical(values=[np.nan, np.nan, np.nan], categories=['a', 'b', 'c']))
    expected.index = index
    expected = DataFrame({'x': expected})
    df = DataFrame({'x': Series(['a', 'b', 'c'], dtype='category')}, index=index)
    tm.assert_frame_equal(df, expected)