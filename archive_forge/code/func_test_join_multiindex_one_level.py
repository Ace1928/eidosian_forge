import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_multiindex_one_level(join_type):
    left = DataFrame(data={'c': 3}, index=MultiIndex.from_tuples([(1, 2)], names=('a', 'b')))
    right = DataFrame(data={'d': 4}, index=MultiIndex.from_tuples([(2,)], names=('b',)))
    result = left.join(right, how=join_type)
    if join_type == 'right':
        expected = DataFrame({'c': [3], 'd': [4]}, index=MultiIndex.from_tuples([(2, 1)], names=['b', 'a']))
    else:
        expected = DataFrame({'c': [3], 'd': [4]}, index=MultiIndex.from_tuples([(1, 2)], names=['a', 'b']))
    tm.assert_frame_equal(result, expected)