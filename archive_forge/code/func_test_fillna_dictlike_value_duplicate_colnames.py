import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
@pytest.mark.parametrize('columns', [['A', 'A', 'B'], ['A', 'A']])
def test_fillna_dictlike_value_duplicate_colnames(self, columns):
    df = DataFrame(np.nan, index=[0, 1], columns=columns)
    with tm.assert_produces_warning(None):
        result = df.fillna({'A': 0})
    expected = df.copy()
    expected['A'] = 0.0
    tm.assert_frame_equal(result, expected)