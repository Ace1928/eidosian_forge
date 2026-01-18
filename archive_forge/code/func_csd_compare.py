import platform
from typing import cast, Literal
import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import ShortTimeFFT
from scipy.signal import csd, get_window, stft, istft
from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext
from scipy.signal._short_time_fft import FFT_MODE_TYPE
from scipy.signal._spectral_py import _spectral_helper, _triage_segments, \
def csd_compare(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean'):
    """Assert that the results from the existing `csd()` and `_csd_wrapper()`
    are close to each other. """
    kw = dict(x=x, y=y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, axis=axis, average=average)
    freqs0, Pxy0 = csd(**kw)
    freqs1, Pxy1 = _csd_wrapper(**kw)
    assert_allclose(freqs1, freqs0)
    assert_allclose(Pxy1, Pxy0)
    assert_allclose(freqs1, freqs0)
    return (freqs0, Pxy0)