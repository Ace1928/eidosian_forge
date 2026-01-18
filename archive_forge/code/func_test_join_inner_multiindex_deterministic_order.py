import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_inner_multiindex_deterministic_order():
    left = DataFrame(data={'e': 5}, index=MultiIndex.from_tuples([(1, 2, 4)], names=('a', 'b', 'd')))
    right = DataFrame(data={'f': 6}, index=MultiIndex.from_tuples([(2, 3)], names=('b', 'c')))
    result = left.join(right, how='inner')
    expected = DataFrame({'e': [5], 'f': [6]}, index=MultiIndex.from_tuples([(1, 2, 4, 3)], names=('a', 'b', 'd', 'c')))
    tm.assert_frame_equal(result, expected)