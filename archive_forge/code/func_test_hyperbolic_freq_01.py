import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_hyperbolic_freq_01(self):
    method = 'hyperbolic'
    t1 = 1.0
    t = np.linspace(0, t1, 10000)
    cases = [[10.0, 1.0], [1.0, 10.0], [-10.0, -1.0], [-1.0, -10.0]]
    for f0, f1 in cases:
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        tf, f = compute_frequency(t, phase)
        expected = chirp_hyperbolic(tf, f0, f1, t1)
        assert_allclose(f, expected)