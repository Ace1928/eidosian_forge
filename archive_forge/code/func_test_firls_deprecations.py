import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_firls_deprecations(self):
    with pytest.deprecated_call(match="argument 'nyq' is deprecated"):
        firls(1, (0, 1), (0, 0), nyq=10)
    with pytest.deprecated_call(match='use keyword arguments'):
        firls(11, [0, 0.1, 0.4, 0.5], [1, 1, 0, 0], None)