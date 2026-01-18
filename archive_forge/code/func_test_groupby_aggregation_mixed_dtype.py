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
def test_groupby_aggregation_mixed_dtype():
    expected = DataFrame({'v1': [5, 5, 7, np.nan, 3, 3, 4, 1], 'v2': [55, 55, 77, np.nan, 33, 33, 44, 11]}, index=MultiIndex.from_tuples([(1, 95), (1, 99), (2, 95), (2, 99), ('big', 'damp'), ('blue', 'dry'), ('red', 'red'), ('red', 'wet')], names=['by1', 'by2']))
    df = DataFrame({'v1': [1, 3, 5, 7, 8, 3, 5, np.nan, 4, 5, 7, 9], 'v2': [11, 33, 55, 77, 88, 33, 55, np.nan, 44, 55, 77, 99], 'by1': ['red', 'blue', 1, 2, np.nan, 'big', 1, 2, 'red', 1, np.nan, 12], 'by2': ['wet', 'dry', 99, 95, np.nan, 'damp', 95, 99, 'red', 99, np.nan, np.nan]})
    g = df.groupby(['by1', 'by2'])
    result = g[['v1', 'v2']].mean()
    tm.assert_frame_equal(result, expected)