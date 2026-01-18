from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def m_num(self) -> int:
    """Number of samples in window `win`.

        Note that the FFT can be oversampled by zero-padding. This is achieved
        by setting the `mfft` property.

        See Also
        --------
        m_num_mid: Center index of window `win`.
        mfft: Length of input for the FFT used - may be larger than `m_num`.
        hop: Time increment in signal samples for sliding window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
    return len(self.win)