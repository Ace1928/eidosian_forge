from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('order', [['a'], ['c'], ['a', 'b'], ['a', 'c'], ['b', 'a'], ['b', 'c'], ['a', 'b', 'c'], ['c', 'a', 'b'], ['c', 'b', 'a'], ['b', 'c', 'a'], ['b', 'a', 'c'], ['b', 'c', 'c']])
@pytest.mark.parametrize('n', range(1, 11))
def test_nlargest_n(self, df_strings, nselect_method, n, order):
    df = df_strings
    if 'b' in order:
        error_msg = f"Column 'b' has dtype (object|string), cannot use method '{nselect_method}' with this dtype"
        with pytest.raises(TypeError, match=error_msg):
            getattr(df, nselect_method)(n, order)
    else:
        ascending = nselect_method == 'nsmallest'
        result = getattr(df, nselect_method)(n, order)
        expected = df.sort_values(order, ascending=ascending).head(n)
        tm.assert_frame_equal(result, expected)