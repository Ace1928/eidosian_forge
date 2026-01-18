import math
from itertools import product
from typing import cast, get_args, Literal
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.fft import fftshift
from scipy.stats import norm as normal_distribution  # type: ignore
from scipy.signal import get_window, welch, stft, istft, spectrogram
from scipy.signal._short_time_fft import FFT_MODE_TYPE, \
from scipy.signal.windows import gaussian
def test_extent():
    """Ensure that the `extent()` method is correct. """
    SFT = ShortTimeFFT(np.ones(32), hop=4, fs=32, fft_mode='onesided')
    assert SFT.extent(100, 'tf', False) == (-0.375, 3.625, 0.0, 17.0)
    assert SFT.extent(100, 'ft', False) == (0.0, 17.0, -0.375, 3.625)
    assert SFT.extent(100, 'tf', True) == (-0.4375, 3.5625, -0.5, 16.5)
    assert SFT.extent(100, 'ft', True) == (-0.5, 16.5, -0.4375, 3.5625)
    SFT = ShortTimeFFT(np.ones(32), hop=4, fs=32, fft_mode='centered')
    assert SFT.extent(100, 'tf', False) == (-0.375, 3.625, -16.0, 15.0)