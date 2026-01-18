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
def test_merge_duplicate_columns_with_suffix_no_warning():
    left = DataFrame([[1, 1, 1], [2, 2, 2]], columns=['a', 'b', 'b'])
    right = DataFrame({'a': [1, 3], 'b': 2})
    result = merge(left, right, on='a')
    expected = DataFrame([[1, 1, 1, 2]], columns=['a', 'b_x', 'b_x', 'b_y'])
    tm.assert_frame_equal(result, expected)