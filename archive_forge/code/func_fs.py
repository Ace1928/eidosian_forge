from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@fs.setter
def fs(self, v: float):
    """Sampling frequency of input signal and of the window.

        The sampling frequency is the inverse of the sampling interval `T`.
        A ``ValueError`` is raised if it is set to a non-positive value.
        """
    if not v > 0:
        raise ValueError(f'Sampling frequency fs={v} must be positive!')
    self._fs = v