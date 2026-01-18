import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('meth', ['pearson', 'kendall', 'spearman'])
def test_corr_int_and_boolean(self, meth):
    pytest.importorskip('scipy')
    df = DataFrame({'a': [True, False], 'b': [1, 0]})
    expected = DataFrame(np.ones((2, 2)), index=['a', 'b'], columns=['a', 'b'])
    result = df.corr(meth)
    tm.assert_frame_equal(result, expected)