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
def test_agg_with_as_index_false_with_list():
    df = DataFrame({'a1': [0, 0, 1], 'a2': [2, 3, 3], 'b': [4, 5, 6]})
    gb = df.groupby(by=['a1', 'a2'], as_index=False)
    result = gb.agg(['sum'])
    expected = DataFrame(data=[[0, 2, 4], [0, 3, 5], [1, 3, 6]], columns=MultiIndex.from_tuples([('a1', ''), ('a2', ''), ('b', 'sum')]))
    tm.assert_frame_equal(result, expected)