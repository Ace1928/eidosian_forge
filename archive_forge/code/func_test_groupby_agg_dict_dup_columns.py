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
def test_groupby_agg_dict_dup_columns():
    df = DataFrame([[1, 2, 3, 4], [1, 3, 4, 5], [2, 4, 5, 6]], columns=['a', 'b', 'c', 'c'])
    gb = df.groupby('a')
    result = gb.agg({'b': 'sum'})
    expected = DataFrame({'b': [5, 4]}, index=Index([1, 2], name='a'))
    tm.assert_frame_equal(result, expected)