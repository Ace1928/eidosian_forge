from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
def k_max(self, n: int) -> int:
    """First sample index after signal end not touched by a time slice.

        `k_max` - 1 is the largest sample index of the slice `p_max` for a
        given input signal of `n` samples.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeFFT: Class this method belongs to.
        """
    return self._post_padding(n)[0]