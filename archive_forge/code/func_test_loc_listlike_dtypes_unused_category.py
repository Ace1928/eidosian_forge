import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_listlike_dtypes_unused_category(self):
    index = CategoricalIndex(['a', 'b', 'a', 'c'], categories=list('abcde'))
    df = DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, index=index)
    res = df.loc[['a', 'b']]
    exp = DataFrame({'A': [1, 3, 2], 'B': [5, 7, 6]}, index=CategoricalIndex(['a', 'a', 'b'], categories=list('abcde')))
    tm.assert_frame_equal(res, exp, check_index_type=True)
    res = df.loc[['a', 'a', 'b']]
    exp = DataFrame({'A': [1, 3, 1, 3, 2], 'B': [5, 7, 5, 7, 6]}, index=CategoricalIndex(['a', 'a', 'a', 'a', 'b'], categories=list('abcde')))
    tm.assert_frame_equal(res, exp, check_index_type=True)
    with pytest.raises(KeyError, match=re.escape("['x'] not in index")):
        df.loc[['a', 'x']]