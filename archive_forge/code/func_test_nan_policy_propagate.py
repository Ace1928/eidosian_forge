import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def test_nan_policy_propagate(self):
    data = [0, 2, 3, -2, np.nan, np.nan]
    assert_array_equal(rankdata(data, nan_policy='propagate'), [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    data = [[0, np.nan, 3], [4, 2, np.nan], [1, 2, 2]]
    assert_array_equal(rankdata(data, axis=0, nan_policy='propagate'), [[1, np.nan, np.nan], [3, np.nan, np.nan], [2, np.nan, np.nan]])
    assert_array_equal(rankdata(data, axis=1, nan_policy='propagate'), [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [1, 2.5, 2.5]])