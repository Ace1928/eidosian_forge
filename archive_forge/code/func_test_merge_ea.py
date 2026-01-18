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
def test_merge_ea(any_numeric_ea_dtype, join_type):
    left = DataFrame({'a': [1, 2, 3], 'b': 1}, dtype=any_numeric_ea_dtype)
    right = DataFrame({'a': [1, 2, 3], 'c': 2}, dtype=any_numeric_ea_dtype)
    result = left.merge(right, how=join_type)
    expected = DataFrame({'a': [1, 2, 3], 'b': 1, 'c': 2}, dtype=any_numeric_ea_dtype)
    tm.assert_frame_equal(result, expected)