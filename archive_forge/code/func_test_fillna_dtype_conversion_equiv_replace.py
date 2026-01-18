import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
@pytest.mark.parametrize('val', ['', 1, np.nan, 1.0])
def test_fillna_dtype_conversion_equiv_replace(self, val):
    df = DataFrame({'A': [1, np.nan], 'B': [1.0, 2.0]})
    expected = df.replace(np.nan, val)
    result = df.fillna(val)
    tm.assert_frame_equal(result, expected)