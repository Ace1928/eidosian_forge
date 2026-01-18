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
def test_groupby_index_object_dtype():
    df = DataFrame({'c0': ['x', 'x', 'x'], 'c1': ['x', 'x', 'y'], 'p': [0, 1, 2]})
    df.index = df.index.astype('O')
    grouped = df.groupby(['c0', 'c1'])
    res = grouped.p.agg(lambda x: all(x > 0))
    expected_index = MultiIndex.from_tuples([('x', 'x'), ('x', 'y')], names=('c0', 'c1'))
    expected = Series([False, True], index=expected_index, name='p')
    tm.assert_series_equal(res, expected)