import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_non_normalized(self):
    """Test windows of small length that are not normalized to 1. See
        the documentation for the Taylor window for more information on
        normalization.
        """
    assert_allclose(windows.taylor(5, 2, 15, norm=False), np.array([0.87508054, 1.04771499, 1.15440894, 1.04771499, 0.87508054]))
    assert_allclose(windows.taylor(6, 2, 15, norm=False), np.array([0.86627793, 1.0, 1.13372207, 1.13372207, 1.0, 0.86627793]))