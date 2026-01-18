import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_kaiser_atten():
    a = kaiser_atten(1, 1.0)
    assert_equal(a, 7.95)
    a = kaiser_atten(2, 1 / np.pi)
    assert_equal(a, 2.285 + 7.95)