from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def fac_magnitude(self) -> float:
    """Factor to multiply the STFT values by to scale each frequency slice
        to a magnitude spectrum.

        It is 1 if attribute ``scaling == 'magnitude'``.
        The window can be scaled to a magnitude spectrum by using the method
        `scale_to`.

        See Also
        --------
        fac_psd: Scaling factor for to a power spectral density spectrum.
        scale_to: Scale window to obtain 'magnitude' or 'psd' scaling.
        scaling: Normalization applied to the window function.
        ShortTimeFFT: Class this property belongs to.
        """
    if self.scaling == 'magnitude':
        return 1
    if self._fac_mag is None:
        self._fac_mag = 1 / abs(sum(self.win))
    return self._fac_mag