import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_fs_nyq(self):
    taps1 = firwin2(80, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    taps2 = firwin2(80, [0.0, 30.0, 60.0], [1.0, 1.0, 0.0], fs=120.0)
    assert_array_almost_equal(taps1, taps2)
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning, "Keyword argument 'nyq'")
        taps2 = firwin2(80, [0.0, 30.0, 60.0], [1.0, 1.0, 0.0], nyq=60.0)
    assert_array_almost_equal(taps1, taps2)