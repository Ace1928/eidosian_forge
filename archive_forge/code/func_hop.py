from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def hop(self) -> int:
    """Time increment in signal samples for sliding window.

        This attribute is read only, since `dual_win` depends on it.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        m_num: Number of samples in window `win`.
        m_num_mid: Center index of window `win`.
        mfft: Length of input for the FFT used - may be larger than `m_num`.
        T: Sampling interval of input signal and of the window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
    return self._hop