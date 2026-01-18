from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def p_min(self) -> int:
    """The smallest possible slice index.

        `p_min` is the index of the left-most slice, where the window still
        sticks into the signal, i.e., has non-zero part for t >= 0.
        `k_min` is the smallest index where the window function of the slice
        `p_min` is non-zero.

        Since, per convention the zeroth slice is centered at t=0,
        `p_min` <= 0 always holds.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeFFT: Class this property belongs to.
        """
    return self._pre_padding()[1]