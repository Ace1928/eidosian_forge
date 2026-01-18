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
def test_merge_equal_cat_dtypes2():
    cat_dtype = CategoricalDtype(categories=['a', 'b', 'c'], ordered=False)
    df1 = DataFrame({'foo': Series(['a', 'b']).astype(cat_dtype), 'left': [1, 2]}).set_index('foo')
    df2 = DataFrame({'foo': Series(['a', 'b', 'c']).astype(cat_dtype), 'right': [3, 2, 1]}).set_index('foo')
    result = df1.merge(df2, left_index=True, right_index=True)
    expected = DataFrame({'left': [1, 2], 'right': [3, 2], 'foo': Series(['a', 'b']).astype(cat_dtype)}).set_index('foo')
    tm.assert_frame_equal(result, expected)