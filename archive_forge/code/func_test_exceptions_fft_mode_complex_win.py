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
@pytest.mark.parametrize('m', ('onesided', 'onesided2X'))
def test_exceptions_fft_mode_complex_win(m: FFT_MODE_TYPE):
    """Verify hat one-sided spectra are not allowed with complex-valued
    windows.

    The reason being, the `rfft` function only accepts real-valued input.
    """
    with pytest.raises(ValueError, match=f"One-sided spectra, i.e., fft_mode='{m}'.*"):
        ShortTimeFFT(np.ones(8) * 1j, hop=4, fs=1, fft_mode=m)
    SFT = ShortTimeFFT(np.ones(8) * 1j, hop=4, fs=1, fft_mode='twosided')
    with pytest.raises(ValueError, match=f"One-sided spectra, i.e., fft_mode='{m}'.*"):
        SFT.fft_mode = m