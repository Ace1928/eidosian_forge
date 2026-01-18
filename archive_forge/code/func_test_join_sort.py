import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
def test_join_sort(self, infer_string):
    with option_context('future.infer_string', infer_string):
        left = DataFrame({'key': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 4]})
        right = DataFrame({'value2': ['a', 'b', 'c']}, index=['bar', 'baz', 'foo'])
        joined = left.join(right, on='key', sort=True)
        expected = DataFrame({'key': ['bar', 'baz', 'foo', 'foo'], 'value': [2, 3, 1, 4], 'value2': ['a', 'b', 'c', 'c']}, index=[1, 2, 0, 3])
        tm.assert_frame_equal(joined, expected)
        joined = left.join(right, on='key', sort=False)
        tm.assert_index_equal(joined.index, Index(range(4)), exact=True)