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
def test_merge_on_left_categoricalindex():
    ci = CategoricalIndex(range(3))
    right = DataFrame({'A': ci, 'B': range(3)})
    left = DataFrame({'C': range(3, 6)})
    res = merge(left, right, left_on=ci, right_on='A')
    expected = merge(left, right, left_on=ci._data, right_on='A')
    tm.assert_frame_equal(res, expected)