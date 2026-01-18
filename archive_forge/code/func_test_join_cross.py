import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize(('input_col', 'output_cols'), [('b', ['a', 'b']), ('a', ['a_x', 'a_y'])])
def test_join_cross(input_col, output_cols):
    left = DataFrame({'a': [1, 3]})
    right = DataFrame({input_col: [3, 4]})
    result = left.join(right, how='cross', lsuffix='_x', rsuffix='_y')
    expected = DataFrame({output_cols[0]: [1, 1, 3, 3], output_cols[1]: [3, 4, 3, 4]})
    tm.assert_frame_equal(result, expected)