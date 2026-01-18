from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('dtype', [None, 'Int64'])
def test_merge_outer_with_NaN(dtype):
    left = DataFrame({'key': [1, 2], 'col1': [1, 2]}, dtype=dtype)
    right = DataFrame({'key': [np.nan, np.nan], 'col2': [3, 4]}, dtype=dtype)
    result = merge(left, right, on='key', how='outer')
    expected = DataFrame({'key': [1, 2, np.nan, np.nan], 'col1': [1, 2, np.nan, np.nan], 'col2': [np.nan, np.nan, 3, 4]}, dtype=dtype)
    tm.assert_frame_equal(result, expected)
    result = merge(right, left, on='key', how='outer')
    expected = DataFrame({'key': [1, 2, np.nan, np.nan], 'col2': [np.nan, np.nan, 3, 4], 'col1': [1, 2, np.nan, np.nan]}, dtype=dtype)
    tm.assert_frame_equal(result, expected)