from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_all_int_arrays(self):
    A = [[0, 1, 0], [0, 0, 1], [-3, -4, -2]]
    B = [[0], [0], [1]]
    C = [[5, 1, 0]]
    D = [[0]]
    num, den = ss2tf(A, B, C, D)
    assert_allclose(num, [[0.0, 0.0, 1.0, 5.0]], rtol=1e-13, atol=1e-14)
    assert_allclose(den, [1.0, 2.0, 4.0, 3.0], rtol=1e-13)