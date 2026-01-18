import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_listlike_dtypes_duplicated_categories_and_codes(self):
    index = CategoricalIndex(['a', 'b', 'a'])
    df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=index)
    res = df.loc[['a', 'b']]
    exp = DataFrame({'A': [1, 3, 2], 'B': [4, 6, 5]}, index=CategoricalIndex(['a', 'a', 'b']))
    tm.assert_frame_equal(res, exp, check_index_type=True)
    res = df.loc[['a', 'a', 'b']]
    exp = DataFrame({'A': [1, 3, 1, 3, 2], 'B': [4, 6, 4, 6, 5]}, index=CategoricalIndex(['a', 'a', 'a', 'a', 'b']))
    tm.assert_frame_equal(res, exp, check_index_type=True)
    with pytest.raises(KeyError, match=re.escape("['x'] not in index")):
        df.loc[['a', 'x']]