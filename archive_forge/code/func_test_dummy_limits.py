from numpy.testing import assert_equal
import numpy as np
def test_dummy_limits(self):
    b, e = dummy_limits(self.d1)
    assert_equal(b, np.array([0, 4, 8]))
    assert_equal(e, np.array([4, 8, 12]))