from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def m_num_mid(self) -> int:
    """Center index of window `win`.

        For odd `m_num`, ``(m_num - 1) / 2`` is returned and
        for even `m_num` (per definition) ``m_num / 2`` is returned.

        See Also
        --------
        m_num: Number of samples in window `win`.
        mfft: Length of input for the FFT used - may be larger than `m_num`.
        hop: ime increment in signal samples for sliding window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
    return self.m_num // 2