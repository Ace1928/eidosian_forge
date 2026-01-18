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
def test_agg_multiple_with_as_index_false_subset_to_a_single_column():
    df = DataFrame({'a': [1, 1, 2], 'b': [3, 4, 5]})
    gb = df.groupby('a', as_index=False)['b']
    result = gb.agg(['sum', 'mean'])
    expected = DataFrame({'a': [1, 2], 'sum': [7, 5], 'mean': [3.5, 5.0]})
    tm.assert_frame_equal(result, expected)