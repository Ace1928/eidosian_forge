import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['Int8', 'Int64'])
def test_melt_ea_dtype(self, dtype):
    df = DataFrame({'a': pd.Series([1, 2], dtype='Int8'), 'b': pd.Series([3, 4], dtype=dtype)})
    result = df.melt()
    expected = DataFrame({'variable': ['a', 'a', 'b', 'b'], 'value': pd.Series([1, 2, 3, 4], dtype=dtype)})
    tm.assert_frame_equal(result, expected)