import numpy as np
from numpy.testing import \
import pytest
from scipy.signal import cont2discrete as c2d
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim, lti
from scipy.signal import tf2ss, impulse, dimpulse, step, dstep
@pytest.mark.parametrize('sys,sample_time,samples_number', cases)
def test_linear_invariant(self, sys, sample_time, samples_number):
    time = np.arange(samples_number) * sample_time
    _, yout_cont, _ = lsim(sys, T=time, U=time)
    _, yout_disc, _ = dlsim(c2d(sys, sample_time, method='foh'), u=time)
    assert_allclose(yout_cont.ravel(), yout_disc.ravel())