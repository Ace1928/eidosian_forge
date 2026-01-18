import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_getitem_periodindex_quarter_string(self):
    pi = PeriodIndex(['2Q05', '3Q05', '4Q05', '1Q06', '2Q06'], freq='Q')
    ser = Series(np.random.default_rng(2).random(len(pi)), index=pi).cumsum()
    assert ser['05Q4'] == ser.iloc[2]