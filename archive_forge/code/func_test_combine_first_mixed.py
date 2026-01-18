from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_mixed(self):
    a = Series(['a', 'b'], index=range(2))
    b = Series(range(2), index=range(2))
    f = DataFrame({'A': a, 'B': b})
    a = Series(['a', 'b'], index=range(5, 7))
    b = Series(range(2), index=range(5, 7))
    g = DataFrame({'A': a, 'B': b})
    exp = DataFrame({'A': list('abab'), 'B': [0, 1, 0, 1]}, index=[0, 1, 5, 6])
    combined = f.combine_first(g)
    tm.assert_frame_equal(combined, exp)