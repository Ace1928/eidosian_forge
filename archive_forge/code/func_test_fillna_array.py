import collections
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_fillna_array(self):
    cat = Categorical(['A', 'B', 'C', None, None])
    other = cat.fillna('C')
    result = cat.fillna(other)
    tm.assert_categorical_equal(result, other)
    assert isna(cat[-1])
    other = np.array(['A', 'B', 'C', 'B', 'A'])
    result = cat.fillna(other)
    expected = Categorical(['A', 'B', 'C', 'B', 'A'], dtype=cat.dtype)
    tm.assert_categorical_equal(result, expected)
    assert isna(cat[-1])