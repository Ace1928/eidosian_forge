import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['first', 'last'])
def test_first_last_with_na_object(method, nulls_fixture):
    groups = DataFrame({'a': [1, 1, 2, 2], 'b': [1, 2, 3, nulls_fixture]}).groupby('a')
    result = getattr(groups, method)()
    if method == 'first':
        values = [1, 3]
    else:
        values = [2, 3]
    values = np.array(values, dtype=result['b'].dtype)
    idx = Index([1, 2], name='a')
    expected = DataFrame({'b': values}, index=idx)
    tm.assert_frame_equal(result, expected)