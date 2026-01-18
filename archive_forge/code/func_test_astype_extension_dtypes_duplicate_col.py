import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['category', 'Int64'])
def test_astype_extension_dtypes_duplicate_col(self, dtype):
    a1 = Series([0, np.nan, 4], name='a')
    a2 = Series([np.nan, 3, 5], name='a')
    df = concat([a1, a2], axis=1)
    result = df.astype(dtype)
    expected = concat([a1.astype(dtype), a2.astype(dtype)], axis=1)
    tm.assert_frame_equal(result, expected)