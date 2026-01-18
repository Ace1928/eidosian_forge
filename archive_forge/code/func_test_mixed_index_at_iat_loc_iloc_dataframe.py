from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_mixed_index_at_iat_loc_iloc_dataframe(self):
    df = DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], columns=['a', 'b', 'c', 1, 2])
    for rowIdx, row in df.iterrows():
        for el, item in row.items():
            assert df.at[rowIdx, el] == df.loc[rowIdx, el] == item
    for row in range(2):
        for i in range(5):
            assert df.iat[row, i] == df.iloc[row, i] == row * 5 + i
    with pytest.raises(KeyError, match='^3$'):
        df.at[0, 3]
    with pytest.raises(KeyError, match='^3$'):
        df.loc[0, 3]