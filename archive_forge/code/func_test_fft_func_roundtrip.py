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
@pytest.mark.parametrize('n', [8, 9])
def test_fft_func_roundtrip(n: int):
    """Test roundtrip `ifft_func(fft_func(x)) == x` for all permutations of
    relevant parameters. """
    np.random.seed(2394795)
    x0 = np.random.rand(n)
    w, h_n = (np.ones(n), 4)
    pp = dict(fft_mode=get_args(FFT_MODE_TYPE), mfft=[None, n, n + 1, n + 2], scaling=[None, 'magnitude', 'psd'], phase_shift=[None, -n + 1, 0, n // 2, n - 1])
    for f_typ, mfft, scaling, phase_shift in product(*pp.values()):
        if f_typ == 'onesided2X' and scaling is None:
            continue
        SFT = ShortTimeFFT(w, h_n, fs=n, fft_mode=f_typ, mfft=mfft, scale_to=scaling, phase_shift=phase_shift)
        X0 = SFT._fft_func(x0)
        x1 = SFT._ifft_func(X0)
        assert_allclose(x0, x1, err_msg='_fft_func() roundtrip failed for ' + f'f_typ={f_typ!r}, mfft={mfft!r}, scaling={scaling!r}, phase_shift={phase_shift!r}')
    SFT = ShortTimeFFT(w, h_n, fs=1)
    SFT._fft_mode = 'invalid_fft'
    with pytest.raises(RuntimeError):
        SFT._fft_func(x0)
    with pytest.raises(RuntimeError):
        SFT._ifft_func(x0)