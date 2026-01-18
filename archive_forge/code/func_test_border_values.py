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
def test_border_values():
    """Ensure that minimum and maximum values of slices are correct."""
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)
    assert SFT.p_min == 0
    assert SFT.k_min == -4
    assert SFT.lower_border_end == (4, 1)
    assert SFT.lower_border_end == (4, 1)
    assert SFT.p_max(10) == 4
    assert SFT.k_max(10) == 16
    assert SFT.upper_border_begin(10) == (4, 2)