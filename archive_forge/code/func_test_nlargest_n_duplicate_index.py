from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('order', [['a', 'b', 'c'], ['c', 'b', 'a'], ['a'], ['b'], ['a', 'b'], ['c', 'b']])
@pytest.mark.parametrize('n', range(1, 6))
def test_nlargest_n_duplicate_index(self, df_duplicates, n, order, request):
    df = df_duplicates
    result = df.nsmallest(n, order)
    expected = df.sort_values(order).head(n)
    tm.assert_frame_equal(result, expected)
    result = df.nlargest(n, order)
    expected = df.sort_values(order, ascending=False).head(n)
    if Version(np.__version__) >= Version('1.25') and (order == ['a'] and n in (1, 2, 3, 4) or (order == ['a', 'b'] and n == 5)):
        request.applymarker(pytest.mark.xfail(reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions', strict=False))
    tm.assert_frame_equal(result, expected)