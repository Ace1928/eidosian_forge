from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def dual_win(self) -> np.ndarray:
    """Canonical dual window.

        A STFT can be interpreted as the input signal being expressed as a
        weighted sum of modulated and time-shifted dual windows. Note that for
        a given window there exist many dual windows. The canonical window is
        the one with the minimal energy (i.e., :math:`L_2` norm).

        `dual_win` has same length as `win`, namely `m_num` samples.

        If the dual window cannot be calculated a ``ValueError`` is raised.
        This attribute is read only and calculated lazily.

        See Also
        --------
        dual_win: Canonical dual window.
        m_num: Number of samples in window `win`.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
    if self._dual_win is None:
        self._dual_win = _calc_dual_canonical_window(self.win, self.hop)
    return self._dual_win