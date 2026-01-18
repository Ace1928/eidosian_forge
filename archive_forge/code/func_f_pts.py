from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def f_pts(self) -> int:
    """Number of points along the frequency axis.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        f: Frequencies values of the STFT.
        mfft: Length of the input for FFT used.
        ShortTimeFFT: Class this property belongs to.
        """
    return self.mfft // 2 + 1 if self.onesided_fft else self.mfft