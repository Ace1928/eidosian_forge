from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('ordered', [True, False])
@pytest.mark.parametrize('category_column,categories,expected_categories', [([False, True, True, False], [True, False], [True, False]), ([2, 1, 1, 2], [1, 2], [1, 2]), (['False', 'True', 'True', 'False'], ['True', 'False'], ['True', 'False'])])
def test_merging_with_bool_or_int_cateorical_column(self, category_column, categories, expected_categories, ordered):
    df1 = DataFrame({'id': [1, 2, 3, 4], 'cat': category_column})
    df1['cat'] = df1['cat'].astype(CategoricalDtype(categories, ordered=ordered))
    df2 = DataFrame({'id': [2, 4], 'num': [1, 9]})
    result = df1.merge(df2)
    expected = DataFrame({'id': [2, 4], 'cat': expected_categories, 'num': [1, 9]})
    expected['cat'] = expected['cat'].astype(CategoricalDtype(categories, ordered=ordered))
    tm.assert_frame_equal(expected, result)