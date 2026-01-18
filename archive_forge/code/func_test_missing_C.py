from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_missing_C(self):
    A, B, C, D = abcd_normalize(A=self.A, B=self.B, D=self.D)
    assert_equal(C.shape[0], D.shape[0])
    assert_equal(C.shape[1], A.shape[0])
    assert_equal(C.shape, (self.D.shape[0], self.A.shape[0]))