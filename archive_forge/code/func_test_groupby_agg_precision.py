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
def test_groupby_agg_precision(any_real_numeric_dtype):
    if any_real_numeric_dtype in tm.ALL_INT_NUMPY_DTYPES:
        max_value = np.iinfo(any_real_numeric_dtype).max
    if any_real_numeric_dtype in tm.FLOAT_NUMPY_DTYPES:
        max_value = np.finfo(any_real_numeric_dtype).max
    if any_real_numeric_dtype in tm.FLOAT_EA_DTYPES:
        max_value = np.finfo(any_real_numeric_dtype.lower()).max
    if any_real_numeric_dtype in tm.ALL_INT_EA_DTYPES:
        max_value = np.iinfo(any_real_numeric_dtype.lower()).max
    df = DataFrame({'key1': ['a'], 'key2': ['b'], 'key3': pd.array([max_value], dtype=any_real_numeric_dtype)})
    arrays = [['a'], ['b']]
    index = MultiIndex.from_arrays(arrays, names=('key1', 'key2'))
    expected = DataFrame({'key3': pd.array([max_value], dtype=any_real_numeric_dtype)}, index=index)
    result = df.groupby(['key1', 'key2']).agg(lambda x: x)
    tm.assert_frame_equal(result, expected)