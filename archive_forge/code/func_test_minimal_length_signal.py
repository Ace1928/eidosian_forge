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
@pytest.mark.parametrize('m_num', [6, 7])
def test_minimal_length_signal(m_num):
    """Verify that the shortest allowed signal works. """
    SFT = ShortTimeFFT(np.ones(m_num), m_num // 2, fs=1)
    n = math.ceil(m_num / 2)
    x = np.ones(n)
    Sx = SFT.stft(x)
    x1 = SFT.istft(Sx, k1=n)
    assert_allclose(x1, x, err_msg=f'Roundtrip minimal length signal (n={n!r})' + f' for {m_num} sample window failed!')
    with pytest.raises(ValueError, match=f'len\\(x\\)={n - 1} must be >= ceil.*'):
        SFT.stft(x[:-1])
    with pytest.raises(ValueError, match=f'S.shape\\[t_axis\\]={Sx.shape[1] - 1} needs to have at least {Sx.shape[1]} slices'):
        SFT.istft(Sx[:, :-1], k1=n)