import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_duplicate_no_raises(self):
    df = DataFrame({'A': [0, 0, 1, 1], 'B': [1, 2, 3, 4]})
    grouped = df.groupby('A').agg(a=('B', 'min'), b=('B', 'min'))
    expected = DataFrame({'a': [1, 3], 'b': [1, 3]}, index=Index([0, 1], name='A'))
    tm.assert_frame_equal(grouped, expected)
    quant50 = functools.partial(np.percentile, q=50)
    quant70 = functools.partial(np.percentile, q=70)
    quant50.__name__ = 'quant50'
    quant70.__name__ = 'quant70'
    test = DataFrame({'col1': ['a', 'a', 'b', 'b', 'b'], 'col2': [1, 2, 3, 4, 5]})
    grouped = test.groupby('col1').agg(quantile_50=('col2', quant50), quantile_70=('col2', quant70))
    expected = DataFrame({'quantile_50': [1.5, 4.0], 'quantile_70': [1.7, 4.4]}, index=Index(['a', 'b'], name='col1'))
    tm.assert_frame_equal(grouped, expected)