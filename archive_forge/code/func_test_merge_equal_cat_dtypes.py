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
@pytest.mark.parametrize('cat_dtype', ['one', 'two'])
@pytest.mark.parametrize('reverse', [True, False])
def test_merge_equal_cat_dtypes(cat_dtype, reverse):
    cat_dtypes = {'one': CategoricalDtype(categories=['a', 'b', 'c'], ordered=False), 'two': CategoricalDtype(categories=['a', 'b', 'c'], ordered=False)}
    df1 = DataFrame({'foo': Series(['a', 'b', 'c']).astype(cat_dtypes['one']), 'left': [1, 2, 3]}).set_index('foo')
    data_foo = ['a', 'b', 'c']
    data_right = [1, 2, 3]
    if reverse:
        data_foo.reverse()
        data_right.reverse()
    df2 = DataFrame({'foo': Series(data_foo).astype(cat_dtypes[cat_dtype]), 'right': data_right}).set_index('foo')
    result = df1.merge(df2, left_index=True, right_index=True)
    expected = DataFrame({'left': [1, 2, 3], 'right': [1, 2, 3], 'foo': Series(['a', 'b', 'c']).astype(cat_dtypes['one'])}).set_index('foo')
    tm.assert_frame_equal(result, expected)