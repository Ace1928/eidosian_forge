from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_getitem_numeric_column_names(self):
    df = DataFrame({0: list('abcd') * 2, 2: np.random.default_rng(2).standard_normal(8), 4: np.random.default_rng(2).standard_normal(8), 6: np.random.default_rng(2).standard_normal(8)})
    result = df.groupby(0)[df.columns[1:3]].mean()
    result2 = df.groupby(0)[[2, 4]].mean()
    expected = df.loc[:, [0, 2, 4]].groupby(0).mean()
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result2, expected)
    with pytest.raises(ValueError, match='Cannot subset columns with a tuple'):
        df.groupby(0)[2, 4].mean()