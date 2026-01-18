import math
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import scipy.fft as fftmodule
from skimage._shared.utils import _supported_float_type
from skimage.data import astronaut, coins
from skimage.filters import butterworth
from skimage.filters._fft_based import _get_nd_butterworth_filter
@pytest.mark.parametrize('squared_butterworth', [False, True])
@pytest.mark.parametrize('high_pass', [False, True])
@pytest.mark.parametrize('order', [6, 10])
@pytest.mark.parametrize('cutoff', [0.2, 0.3])
def test_butterworth_cutoff(cutoff, order, high_pass, squared_butterworth):
    wfilt = _get_nd_butterworth_filter(shape=(512, 512), factor=cutoff, order=order, high_pass=high_pass, real=False, squared_butterworth=squared_butterworth)
    wfilt_profile = np.abs(wfilt[0])
    tol = 0.3 / order
    if high_pass:
        assert abs(wfilt_profile[wfilt_profile.size // 2] - 1.0) < tol
    else:
        assert abs(wfilt_profile[0] - 1.0) < tol
    f_cutoff = int(cutoff * wfilt.shape[0])
    if squared_butterworth:
        assert abs(wfilt_profile[f_cutoff] - 0.5) < tol
    else:
        assert abs(wfilt_profile[f_cutoff] - 1 / math.sqrt(2)) < tol