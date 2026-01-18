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
def test_exceptions_properties_methods():
    """Verify that exceptions get raised when setting properties or calling
    method of ShortTimeFFT to/with invalid values."""
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)
    with pytest.raises(ValueError, match='Sampling interval T=-1 must be ' + 'positive!'):
        SFT.T = -1
    with pytest.raises(ValueError, match='Sampling frequency fs=-1 must be ' + 'positive!'):
        SFT.fs = -1
    with pytest.raises(ValueError, match="fft_mode='invalid_typ' not in " + "\\('twosided', 'centered', " + "'onesided', 'onesided2X'\\)!"):
        SFT.fft_mode = 'invalid_typ'
    with pytest.raises(ValueError, match='For scaling is None, ' + "fft_mode='onesided2X' is invalid.*"):
        SFT.fft_mode = 'onesided2X'
    with pytest.raises(ValueError, match='Attribute mfft=7 needs to be ' + 'at least the window length.*'):
        SFT.mfft = 7
    with pytest.raises(ValueError, match="scaling='invalid' not in.*"):
        SFT.scale_to('invalid')
    with pytest.raises(ValueError, match='phase_shift=3.0 has the unit .*'):
        SFT.phase_shift = 3.0
    with pytest.raises(ValueError, match='-mfft < phase_shift < mfft ' + 'does not hold.*'):
        SFT.phase_shift = 2 * SFT.mfft
    with pytest.raises(ValueError, match="Parameter padding='invalid' not.*"):
        g = SFT._x_slices(np.zeros(16), k_off=0, p0=0, p1=1, padding='invalid')
        next(g)
    with pytest.raises(ValueError, match="Trend type must be 'linear' " + "or 'constant'"):
        SFT.stft_detrend(np.zeros(16), detr='invalid')
    with pytest.raises(ValueError, match='Parameter detr=nan is not a str, ' + 'function or None!'):
        SFT.stft_detrend(np.zeros(16), detr=np.nan)
    with pytest.raises(ValueError, match='Invalid Parameter p0=0, p1=200.*'):
        SFT.p_range(100, 0, 200)
    with pytest.raises(ValueError, match='f_axis=0 may not be equal to ' + 't_axis=0!'):
        SFT.istft(np.zeros((SFT.f_pts, 2)), t_axis=0, f_axis=0)
    with pytest.raises(ValueError, match='S.shape\\[f_axis\\]=2 must be equal' + ' to self.f_pts=5.*'):
        SFT.istft(np.zeros((2, 2)))
    with pytest.raises(ValueError, match='S.shape\\[t_axis\\]=1 needs to have' + ' at least 2 slices.*'):
        SFT.istft(np.zeros((SFT.f_pts, 1)))
    with pytest.raises(ValueError, match='.*\\(k1=100\\) <= \\(k_max=12\\) ' + 'is false!$'):
        SFT.istft(np.zeros((SFT.f_pts, 3)), k1=100)
    with pytest.raises(ValueError, match='\\(k1=1\\) - \\(k0=0\\) = 1 has to ' + 'be at least.* length 4!'):
        SFT.istft(np.zeros((SFT.f_pts, 3)), k0=0, k1=1)
    with pytest.raises(ValueError, match="Parameter axes_seq='invalid' " + "not in \\['tf', 'ft'\\]!"):
        SFT.extent(n=100, axes_seq='invalid')
    with pytest.raises(ValueError, match='Attribute fft_mode=twosided must.*'):
        SFT.fft_mode = 'twosided'
        SFT.extent(n=100)