import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
def test_aggregate_float64_no_int64():
    df = DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 2, 4, 5], 'c': [1, 2, 3, 4, 5]})
    expected = DataFrame({'a': [1, 2.5, 4, 5]}, index=[1, 2, 4, 5])
    expected.index.name = 'b'
    result = df.groupby('b')[['a']].mean()
    tm.assert_frame_equal(result, expected)
    expected = DataFrame({'a': [1, 2.5, 4, 5], 'c': [1, 2.5, 4, 5]}, index=[1, 2, 4, 5])
    expected.index.name = 'b'
    result = df.groupby('b')[['a', 'c']].mean()
    tm.assert_frame_equal(result, expected)