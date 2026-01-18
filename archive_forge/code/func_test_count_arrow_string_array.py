from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_count_arrow_string_array(any_string_dtype):
    pytest.importorskip('pyarrow')
    df = DataFrame({'a': [1, 2, 3], 'b': Series(['a', 'b', 'a'], dtype=any_string_dtype)})
    result = df.groupby('a').count()
    expected = DataFrame({'b': 1}, index=Index([1, 2, 3], name='a'))
    tm.assert_frame_equal(result, expected)