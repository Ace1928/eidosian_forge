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
def test__calc_dual_canonical_window_roundtrip():
    """Test dual window calculation with a round trip to verify duality.

    Note that this works only for canonical window pairs (having minimal
    energy) like a Gaussian.

    The window is the same as in the example of `from ShortTimeFFT.from_dual`.
    """
    win = gaussian(51, std=10, sym=True)
    d_win = _calc_dual_canonical_window(win, 10)
    win2 = _calc_dual_canonical_window(d_win, 10)
    assert_allclose(win2, win)