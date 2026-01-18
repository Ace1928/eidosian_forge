import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_on_inner(self):
    df = DataFrame({'key': ['a', 'a', 'd', 'b', 'b', 'c']})
    df2 = DataFrame({'value': [0, 1]}, index=['a', 'b'])
    joined = df.join(df2, on='key', how='inner')
    expected = df.join(df2, on='key')
    expected = expected[expected['value'].notna()]
    tm.assert_series_equal(joined['key'], expected['key'])
    tm.assert_series_equal(joined['value'], expected['value'], check_dtype=False)
    tm.assert_index_equal(joined.index, expected.index)