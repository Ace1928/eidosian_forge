from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_repr(self):
    a = Series(Categorical([1, 2, 3, 4]))
    exp = '0    1\n1    2\n2    3\n3    4\n' + 'dtype: category\nCategories (4, int64): [1, 2, 3, 4]'
    assert exp == a.__str__()
    a = Series(Categorical(['a', 'b'] * 25))
    exp = '0     a\n1     b\n' + '     ..\n' + '48    a\n49    b\n' + "Length: 50, dtype: category\nCategories (2, object): ['a', 'b']"
    with option_context('display.max_rows', 5):
        assert exp == repr(a)
    levs = list('abcdefghijklmnopqrstuvwxyz')
    a = Series(Categorical(['a', 'b'], categories=levs, ordered=True))
    exp = '0    a\n1    b\n' + "dtype: category\nCategories (26, object): ['a' < 'b' < 'c' < 'd' ... 'w' < 'x' < 'y' < 'z']"
    assert exp == a.__str__()