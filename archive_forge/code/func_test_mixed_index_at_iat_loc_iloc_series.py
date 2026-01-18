from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_mixed_index_at_iat_loc_iloc_series(self):
    s = Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 1, 2])
    for el, item in s.items():
        assert s.at[el] == s.loc[el] == item
    for i in range(len(s)):
        assert s.iat[i] == s.iloc[i] == i + 1
    with pytest.raises(KeyError, match='^4$'):
        s.at[4]
    with pytest.raises(KeyError, match='^4$'):
        s.loc[4]