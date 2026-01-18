from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
def p_max(self, n: int) -> int:
    """Index of first non-overlapping upper time slice for `n` sample
        input.

        Note that center point t[p_max] = (p_max(n)-1) * `delta_t` is typically
        larger than last time index t[n-1] == (`n`-1) * `T`. The upper border
        of samples indexes covered by the window slices is given by `k_max`.
        Furthermore, `p_max` does not denote the number of slices `p_num` since
        `p_min` is typically less than zero.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        p_min: The smallest possible slice index.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeFFT: Class this method belongs to.
        """
    return self._post_padding(n)[1]