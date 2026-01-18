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
def test_agg_split_block():
    df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'], 'key2': ['one', 'two', 'one', 'two', 'one'], 'key3': ['three', 'three', 'three', 'six', 'six']})
    result = df.groupby('key1').min()
    expected = DataFrame({'key2': ['one', 'one'], 'key3': ['six', 'six']}, index=Index(['a', 'b'], name='key1'))
    tm.assert_frame_equal(result, expected)