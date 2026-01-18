import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_input_modyfication(self):
    freq1 = np.array([0.0, 0.5, 0.5, 1.0])
    freq2 = np.array(freq1)
    firwin2(80, freq1, [1.0, 1.0, 0.0, 0.0])
    assert_equal(freq1, freq2)