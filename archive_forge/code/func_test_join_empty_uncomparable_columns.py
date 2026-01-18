import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_empty_uncomparable_columns():
    df1 = DataFrame()
    df2 = DataFrame(columns=['test'])
    df3 = DataFrame(columns=['foo', ('bar', 'baz')])
    result = df1 + df2
    expected = DataFrame(columns=['test'])
    tm.assert_frame_equal(result, expected)
    result = df2 + df3
    expected = DataFrame(columns=[('bar', 'baz'), 'foo', 'test'])
    tm.assert_frame_equal(result, expected)
    result = df1 + df3
    expected = DataFrame(columns=[('bar', 'baz'), 'foo'])
    tm.assert_frame_equal(result, expected)