from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_empty_columns():
    left = DataFrame(columns=['a', 'b'])
    right = DataFrame(columns=['a', 'c'])
    result = left.combine_first(right)
    expected = DataFrame(columns=['a', 'b', 'c'])
    tm.assert_frame_equal(result, expected)