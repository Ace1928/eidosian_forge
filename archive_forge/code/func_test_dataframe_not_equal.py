import numpy as np
from pandas import (
import pandas._testing as tm
def test_dataframe_not_equal(self):
    df1 = DataFrame({'a': [1, 2], 'b': ['s', 'd']})
    df2 = DataFrame({'a': ['s', 'd'], 'b': [1, 2]})
    assert df1.equals(df2) is False