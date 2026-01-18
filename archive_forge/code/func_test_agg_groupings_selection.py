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
def test_agg_groupings_selection():
    df = DataFrame({'a': [1, 1, 2], 'b': [3, 3, 4], 'c': [5, 6, 7]})
    gb = df.groupby(['a', 'b'])
    selected_gb = gb[['b', 'c']]
    result = selected_gb.agg(lambda x: x.sum())
    index = MultiIndex(levels=[[1, 2], [3, 4]], codes=[[0, 1], [0, 1]], names=['a', 'b'])
    expected = DataFrame({'b': [6, 4], 'c': [11, 7]}, index=index)
    tm.assert_frame_equal(result, expected)