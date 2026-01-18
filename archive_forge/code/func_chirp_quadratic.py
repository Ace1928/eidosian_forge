import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def chirp_quadratic(t, f0, f1, t1, vertex_zero=True):
    if vertex_zero:
        f = f0 + (f1 - f0) * t ** 2 / t1 ** 2
    else:
        f = f1 - (f1 - f0) * (t1 - t) ** 2 / t1 ** 2
    return f