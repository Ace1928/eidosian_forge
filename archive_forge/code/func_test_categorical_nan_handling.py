from datetime import timedelta
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_nan_handling(self):
    s = Series(Categorical(['a', 'b', np.nan, 'a']))
    tm.assert_index_equal(s.cat.categories, Index(['a', 'b']))
    tm.assert_numpy_array_equal(s.values.codes, np.array([0, 1, -1, 0], dtype=np.int8))