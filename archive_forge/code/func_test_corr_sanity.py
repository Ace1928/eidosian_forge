import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
def test_corr_sanity():
    df = DataFrame(np.array([[0.87024726, 0.18505595], [0.64355431, 0.3091617], [0.92372966, 0.50552513], [0.00203756, 0.04520709], [0.84780328, 0.33394331], [0.78369152, 0.63919667]]))
    res = df[0].rolling(5, center=True).corr(df[1])
    assert all((np.abs(np.nan_to_num(x)) <= 1 for x in res))
    df = DataFrame(np.random.default_rng(2).random((30, 2)))
    res = df[0].rolling(5, center=True).corr(df[1])
    assert all((np.abs(np.nan_to_num(x)) <= 1 for x in res))