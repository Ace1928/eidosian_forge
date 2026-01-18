import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_even_highpass_raises_value_error(self):
    """Test that attempt to create a highpass filter with an even number
        of taps raises a ValueError exception."""
    assert_raises(ValueError, firwin, 40, 0.5, pass_zero=False)
    assert_raises(ValueError, firwin, 40, [0.25, 0.5])