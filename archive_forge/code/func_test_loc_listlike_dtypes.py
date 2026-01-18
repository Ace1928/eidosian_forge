import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_listlike_dtypes(self):
    index = CategoricalIndex(['a', 'b', 'c'])
    df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=index)
    res = df.loc[['a', 'b']]
    exp_index = CategoricalIndex(['a', 'b'], categories=index.categories)
    exp = DataFrame({'A': [1, 2], 'B': [4, 5]}, index=exp_index)
    tm.assert_frame_equal(res, exp, check_index_type=True)
    res = df.loc[['a', 'a', 'b']]
    exp_index = CategoricalIndex(['a', 'a', 'b'], categories=index.categories)
    exp = DataFrame({'A': [1, 1, 2], 'B': [4, 4, 5]}, index=exp_index)
    tm.assert_frame_equal(res, exp, check_index_type=True)
    with pytest.raises(KeyError, match=re.escape("['x'] not in index")):
        df.loc[['a', 'x']]