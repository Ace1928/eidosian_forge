import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('other_column', [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])])
def test_cov_nullable_integer(self, other_column):
    data = DataFrame({'a': pd.array([1, 2, None]), 'b': other_column})
    result = data.cov()
    arr = np.array([[0.5, 0.5], [0.5, 1.0]])
    expected = DataFrame(arr, columns=['a', 'b'], index=['a', 'b'])
    tm.assert_frame_equal(result, expected)