from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@mfft.setter
def mfft(self, n_: int):
    """Setter for the length of FFT utilized.

        See the property `mfft` for further details.
        """
    if not n_ >= self.m_num:
        raise ValueError(f'Attribute mfft={n_} needs to be at least the ' + f'window length m_num={self.m_num}!')
    self._mfft = n_