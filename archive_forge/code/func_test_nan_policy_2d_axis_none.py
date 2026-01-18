import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def test_nan_policy_2d_axis_none(self):
    data = [[0, np.nan, 3], [4, 2, np.nan], [1, 2, 2]]
    assert_array_equal(rankdata(data, axis=None, nan_policy='omit'), [1.0, np.nan, 6.0, 7.0, 4.0, np.nan, 2.0, 4.0, 4.0])
    assert_array_equal(rankdata(data, axis=None, nan_policy='propagate'), [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])