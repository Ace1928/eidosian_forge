from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_categorical_with_coercion(self):
    factor = Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'])
    s = Series(factor, name='A')
    assert s.dtype == 'category'
    assert len(s) == len(factor)
    df = DataFrame({'A': factor})
    result = df['A']
    tm.assert_series_equal(result, s)
    result = df.iloc[:, 0]
    tm.assert_series_equal(result, s)
    assert len(df) == len(factor)
    df = DataFrame({'A': s})
    result = df['A']
    tm.assert_series_equal(result, s)
    assert len(df) == len(factor)
    df = DataFrame({'A': s, 'B': s, 'C': 1})
    result1 = df['A']
    result2 = df['B']
    tm.assert_series_equal(result1, s)
    tm.assert_series_equal(result2, s, check_names=False)
    assert result2.name == 'B'
    assert len(df) == len(factor)