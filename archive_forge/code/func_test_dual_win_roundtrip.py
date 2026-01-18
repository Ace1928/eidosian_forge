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
def test_dual_win_roundtrip():
    """Verify the duality of `win` and `dual_win`.

    Note that this test does not work for arbitrary windows, since dual windows
    are not unique. It always works for invertible STFTs if the windows do not
    overlap.
    """
    kw = dict(hop=4, fs=1, fft_mode='twosided', mfft=8, scale_to=None, phase_shift=2)
    SFT0 = ShortTimeFFT(np.ones(4), **kw)
    SFT1 = ShortTimeFFT.from_dual(SFT0.dual_win, **kw)
    assert_allclose(SFT1.dual_win, SFT0.win)