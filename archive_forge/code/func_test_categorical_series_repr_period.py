from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_series_repr_period(self):
    idx = period_range('2011-01-01 09:00', freq='H', periods=5)
    s = Series(Categorical(idx))
    exp = '0    2011-01-01 09:00\n1    2011-01-01 10:00\n2    2011-01-01 11:00\n3    2011-01-01 12:00\n4    2011-01-01 13:00\ndtype: category\nCategories (5, period[H]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,\n                            2011-01-01 13:00]'
    assert repr(s) == exp
    idx = period_range('2011-01', freq='M', periods=5)
    s = Series(Categorical(idx))
    exp = '0    2011-01\n1    2011-02\n2    2011-03\n3    2011-04\n4    2011-05\ndtype: category\nCategories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]'
    assert repr(s) == exp