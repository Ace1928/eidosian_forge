import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def chirp_linear(t, f0, f1, t1):
    f = f0 + (f1 - f0) * t / t1
    return f