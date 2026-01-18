import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_bad_cutoff(self):
    """Test that invalid cutoff argument raises ValueError."""
    assert_raises(ValueError, firwin, 99, -0.5)
    assert_raises(ValueError, firwin, 99, 1.5)
    assert_raises(ValueError, firwin, 99, [0, 0.5])
    assert_raises(ValueError, firwin, 99, [0.5, 1])
    assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.2])
    assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.5])
    assert_raises(ValueError, firwin, 99, [])
    assert_raises(ValueError, firwin, 99, [[0.1, 0.2], [0.3, 0.4]])
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning, "Keyword argument 'nyq'")
        assert_raises(ValueError, firwin, 99, 50.0, nyq=40)
        assert_raises(ValueError, firwin, 99, [10, 20, 30], nyq=25)
    assert_raises(ValueError, firwin, 99, 50.0, fs=80)
    assert_raises(ValueError, firwin, 99, [10, 20, 30], fs=50)