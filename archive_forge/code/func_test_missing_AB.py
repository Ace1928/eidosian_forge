from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_missing_AB(self):
    A, B, C, D = abcd_normalize(C=self.C, D=self.D)
    assert_equal(A.shape[0], A.shape[1])
    assert_equal(A.shape[0], B.shape[0])
    assert_equal(B.shape[1], D.shape[1])
    assert_equal(A.shape, (self.C.shape[1], self.C.shape[1]))
    assert_equal(B.shape, (self.C.shape[1], self.D.shape[1]))