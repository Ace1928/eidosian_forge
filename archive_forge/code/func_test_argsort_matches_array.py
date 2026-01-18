import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.indexes.common import Base
def test_argsort_matches_array(self, simple_index):
    idx = simple_index
    idx = idx.insert(1, pd.NaT)
    result = idx.argsort()
    expected = idx._data.argsort()
    tm.assert_numpy_array_equal(result, expected)