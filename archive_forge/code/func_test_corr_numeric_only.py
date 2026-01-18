import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('meth', ['pearson', 'kendall', 'spearman'])
@pytest.mark.parametrize('numeric_only', [True, False])
def test_corr_numeric_only(self, meth, numeric_only):
    pytest.importorskip('scipy')
    df = DataFrame({'a': [1, 0], 'b': [1, 0], 'c': ['x', 'y']})
    expected = DataFrame(np.ones((2, 2)), index=['a', 'b'], columns=['a', 'b'])
    if numeric_only:
        result = df.corr(meth, numeric_only=numeric_only)
        tm.assert_frame_equal(result, expected)
    else:
        with pytest.raises(ValueError, match='could not convert string to float'):
            df.corr(meth, numeric_only=numeric_only)