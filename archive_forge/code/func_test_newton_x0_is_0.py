import numpy as np
from numpy.testing import assert_almost_equal
from pytest import raises as assert_raises
import scipy.optimize
def test_newton_x0_is_0(self):
    tgt = 1
    res = scipy.optimize.newton(lambda x: x - 1, 0)
    assert_almost_equal(res, tgt)