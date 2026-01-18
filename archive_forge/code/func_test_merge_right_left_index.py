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
def test_merge_right_left_index():
    left = DataFrame({'x': [1, 1], 'z': ['foo', 'foo']})
    right = DataFrame({'x': [1, 1], 'z': ['foo', 'foo']})
    result = merge(left, right, how='right', left_index=True, right_on='x')
    expected = DataFrame({'x': [1, 1], 'x_x': [1, 1], 'z_x': ['foo', 'foo'], 'x_y': [1, 1], 'z_y': ['foo', 'foo']})
    tm.assert_frame_equal(result, expected)