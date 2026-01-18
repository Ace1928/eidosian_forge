import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multiindex_loc_one_dimensional_tuple(self, frame_or_series):
    mi = MultiIndex.from_tuples([('a', 'A'), ('b', 'A')])
    obj = frame_or_series([1, 2], index=mi)
    obj.loc['a',] = 0
    expected = frame_or_series([0, 2], index=mi)
    tm.assert_equal(obj, expected)