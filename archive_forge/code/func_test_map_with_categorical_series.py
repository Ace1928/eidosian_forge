import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_with_categorical_series():
    a = Index([1, 2, 3, 4])
    b = Series(['even', 'odd', 'even', 'odd'], dtype='category')
    c = Series(['even', 'odd', 'even', 'odd'])
    exp = CategoricalIndex(['odd', 'even', 'odd', np.nan])
    tm.assert_index_equal(a.map(b), exp)
    exp = Index(['odd', 'even', 'odd', np.nan])
    tm.assert_index_equal(a.map(c), exp)