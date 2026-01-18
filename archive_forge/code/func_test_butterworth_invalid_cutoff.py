import math
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import scipy.fft as fftmodule
from skimage._shared.utils import _supported_float_type
from skimage.data import astronaut, coins
from skimage.filters import butterworth
from skimage.filters._fft_based import _get_nd_butterworth_filter
@pytest.mark.parametrize('cutoff', [-0.01, 0.51])
def test_butterworth_invalid_cutoff(cutoff):
    with pytest.raises(ValueError):
        butterworth(np.ones((4, 4)), cutoff_frequency_ratio=cutoff)