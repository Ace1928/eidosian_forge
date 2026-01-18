import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def test_big_tie(self):
    for n in [10000, 100000, 1000000]:
        data = np.ones(n, dtype=int)
        r = rankdata(data)
        expected_rank = 0.5 * (n + 1)
        assert_array_equal(r, expected_rank * data, 'test failed with n=%d' % n)