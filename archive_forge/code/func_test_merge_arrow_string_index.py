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
def test_merge_arrow_string_index(any_string_dtype):
    pytest.importorskip('pyarrow')
    left = DataFrame({'a': ['a', 'b']}, dtype=any_string_dtype)
    right = DataFrame({'b': 1}, index=Index(['a', 'c'], dtype=any_string_dtype))
    result = left.merge(right, left_on='a', right_index=True, how='left')
    expected = DataFrame({'a': Series(['a', 'b'], dtype=any_string_dtype), 'b': [1, np.nan]})
    tm.assert_frame_equal(result, expected)