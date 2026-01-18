import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
def test_chebys(self):
    S0 = orth.chebys(0)
    S1 = orth.chebys(1)
    S2 = orth.chebys(2)
    S3 = orth.chebys(3)
    S4 = orth.chebys(4)
    S5 = orth.chebys(5)
    assert_array_almost_equal(S0.c, [1], 13)
    assert_array_almost_equal(S1.c, [1, 0], 13)
    assert_array_almost_equal(S2.c, [1, 0, -1], 13)
    assert_array_almost_equal(S3.c, [1, 0, -2, 0], 13)
    assert_array_almost_equal(S4.c, [1, 0, -3, 0, 1], 13)
    assert_array_almost_equal(S5.c, [1, 0, -4, 0, 3, 0], 13)