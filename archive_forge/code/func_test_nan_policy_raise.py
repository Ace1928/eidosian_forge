import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def test_nan_policy_raise(self):
    data = [0, 2, 3, -2, np.nan, np.nan]
    with pytest.raises(ValueError, match='The input contains nan'):
        rankdata(data, nan_policy='raise')
    data = [[0, np.nan, 3], [4, 2, np.nan], [np.nan, 2, 2]]
    with pytest.raises(ValueError, match='The input contains nan'):
        rankdata(data, axis=0, nan_policy='raise')
    with pytest.raises(ValueError, match='The input contains nan'):
        rankdata(data, axis=1, nan_policy='raise')