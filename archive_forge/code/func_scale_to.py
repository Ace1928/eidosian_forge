from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
def scale_to(self, scaling: Literal['magnitude', 'psd']):
    """Scale window to obtain 'magnitude' or 'psd' scaling for the STFT.

        The window of a 'magnitude' spectrum has an integral of one, i.e., unit
        area for non-negative windows. This ensures that absolute the values of
        spectrum does not change if the length of the window changes (given
        the input signal is stationary).

        To represent the power spectral density ('psd') for varying length
        windows the area of the absolute square of the window needs to be
        unity.

        The `scaling` property shows the current scaling. The properties
        `fac_magnitude` and `fac_psd` show the scaling factors required to
        scale the STFT values to a magnitude or a psd spectrum.

        This method is called, if the initializer parameter `scale_to` is set.

        See Also
        --------
        fac_magnitude: Scaling factor for to  a magnitude spectrum.
        fac_psd: Scaling factor for to  a power spectral density spectrum.
        fft_mode: Mode of utilized FFT
        scaling: Normalization applied to the window function.
        ShortTimeFFT: Class this method belongs to.
        """
    if scaling not in (scaling_values := {'magnitude', 'psd'}):
        raise ValueError(f'scaling={scaling!r} not in {scaling_values}!')
    if self._scaling == scaling:
        return
    s_fac = self.fac_psd if scaling == 'psd' else self.fac_magnitude
    self._win = self._win * s_fac
    if self._dual_win is not None:
        self._dual_win = self._dual_win / s_fac
    self._fac_mag, self._fac_psd = (None, None)
    self._scaling = scaling