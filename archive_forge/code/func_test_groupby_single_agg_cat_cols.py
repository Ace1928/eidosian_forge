import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('grp_col_dict, exp_data', [({'nr': 'min', 'cat_ord': 'min'}, {'nr': [1, 5], 'cat_ord': ['a', 'c']}), ({'cat_ord': 'min'}, {'cat_ord': ['a', 'c']}), ({'nr': 'min'}, {'nr': [1, 5]})])
def test_groupby_single_agg_cat_cols(grp_col_dict, exp_data):
    input_df = DataFrame({'nr': [1, 2, 3, 4, 5, 6, 7, 8], 'cat_ord': list('aabbccdd'), 'cat': list('aaaabbbb')})
    input_df = input_df.astype({'cat': 'category', 'cat_ord': 'category'})
    input_df['cat_ord'] = input_df['cat_ord'].cat.as_ordered()
    result_df = input_df.groupby('cat', observed=False).agg(grp_col_dict)
    cat_index = pd.CategoricalIndex(['a', 'b'], categories=['a', 'b'], ordered=False, name='cat', dtype='category')
    expected_df = DataFrame(data=exp_data, index=cat_index)
    if 'cat_ord' in expected_df:
        dtype = input_df['cat_ord'].dtype
        expected_df['cat_ord'] = expected_df['cat_ord'].astype(dtype)
    tm.assert_frame_equal(result_df, expected_df)