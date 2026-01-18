from datetime import datetime
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_append_names_dont_match():
    midx = MultiIndex.from_arrays([[1, 2], [3, 4]], names=['a', 'b'])
    midx2 = MultiIndex.from_arrays([[3], [5]], names=['x', 'y'])
    result = midx.append(midx2)
    expected = MultiIndex.from_arrays([[1, 2, 3], [3, 4, 5]], names=None)
    tm.assert_index_equal(result, expected)