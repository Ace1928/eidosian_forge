from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
def extent(self, n: int, axes_seq: Literal['tf', 'ft']='tf', center_bins: bool=False) -> tuple[float, float, float, float]:
    """Return minimum and maximum values time-frequency values.

        A tuple with four floats  ``(t0, t1, f0, f1)`` for 'tf' and
        ``(f0, f1, t0, t1)`` for 'ft') is returned describing the corners
        of the time-frequency domain of the `~ShortTimeFFT.stft`.
        That tuple can be passed to `matplotlib.pyplot.imshow` as a parameter
        with the same name.

        Parameters
        ----------
        n : int
            Number of samples in input signal.
        axes_seq : {'tf', 'ft'}
            Return time extent first and then frequency extent or vice-versa.
        center_bins: bool
            If set (default ``False``), the values of the time slots and
            frequency bins are moved from the side the middle. This is useful,
            when plotting the `~ShortTimeFFT.stft` values as step functions,
            i.e., with no interpolation.

        See Also
        --------
        :func:`matplotlib.pyplot.imshow`: Display data as an image.
        :class:`scipy.signal.ShortTimeFFT`: Class this method belongs to.
        """
    if axes_seq not in ('tf', 'ft'):
        raise ValueError(f"Parameter axes_seq={axes_seq!r} not in ['tf', 'ft']!")
    if self.onesided_fft:
        q0, q1 = (0, self.f_pts)
    elif self.fft_mode == 'centered':
        q0 = -self.mfft // 2
        q1 = self.mfft // 2 - 1 if self.mfft % 2 == 0 else self.mfft // 2
    else:
        raise ValueError(f'Attribute fft_mode={self.fft_mode} must be ' + "in ['centered', 'onesided', 'onesided2X']")
    p0, p1 = (self.p_min, self.p_max(n))
    if center_bins:
        t0, t1 = (self.delta_t * (p0 - 0.5), self.delta_t * (p1 - 0.5))
        f0, f1 = (self.delta_f * (q0 - 0.5), self.delta_f * (q1 - 0.5))
    else:
        t0, t1 = (self.delta_t * p0, self.delta_t * p1)
        f0, f1 = (self.delta_f * q0, self.delta_f * q1)
    return (t0, t1, f0, f1) if axes_seq == 'tf' else (f0, f1, t0, t1)