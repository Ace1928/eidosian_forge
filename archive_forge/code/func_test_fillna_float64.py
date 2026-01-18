import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_fillna_float64(self):
    index_cls = Index
    idx = Index([1.0, np.nan, 3.0], dtype=float, name='x')
    exp = Index([1.0, 0.1, 3.0], name='x')
    tm.assert_index_equal(idx.fillna(0.1), exp, exact=True)
    exp = index_cls([1.0, 2.0, 3.0], name='x')
    tm.assert_index_equal(idx.fillna(2), exp)
    exp = Index([1.0, 'obj', 3.0], name='x')
    tm.assert_index_equal(idx.fillna('obj'), exp, exact=True)