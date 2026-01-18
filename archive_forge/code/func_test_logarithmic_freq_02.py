import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_logarithmic_freq_02(self):
    method = 'logarithmic'
    f0 = 200.0
    f1 = 100.0
    t1 = 10.0
    t = np.linspace(0, t1, 10000)
    phase = waveforms._chirp_phase(t, f0, t1, f1, method)
    tf, f = compute_frequency(t, phase)
    abserr = np.max(np.abs(f - chirp_geometric(tf, f0, f1, t1)))
    assert_(abserr < 1e-06)