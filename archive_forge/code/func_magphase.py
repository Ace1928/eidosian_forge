import warnings
import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import scipy.interpolate
from numba import jit
from . import convert
from .fft import get_fftlib
from .audio import resample
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..filters import get_window, semitone_filterbank
from ..filters import window_sumsquare
from numpy.typing import DTypeLike
from typing import Any, Callable, Optional, Tuple, List, Union, overload
from typing_extensions import Literal
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
def magphase(D: np.ndarray, *, power: float=1) -> Tuple[np.ndarray, np.ndarray]:
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that ``D = S * P``.

    Parameters
    ----------
    D : np.ndarray [shape=(..., d, t), dtype=complex]
        complex-valued spectrogram
    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    Returns
    -------
    D_mag : np.ndarray [shape=(..., d, t), dtype=real]
        magnitude of ``D``, raised to ``power``
    D_phase : np.ndarray [shape=(..., d, t), dtype=complex]
        ``exp(1.j * phi)`` where ``phi`` is the phase of ``D``

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> D = librosa.stft(y)
    >>> magnitude, phase = librosa.magphase(D)
    >>> magnitude
    array([[5.395e-03, 3.332e-03, ..., 9.862e-07, 1.201e-05],
           [3.244e-03, 2.690e-03, ..., 9.536e-07, 1.201e-05],
           ...,
           [7.523e-05, 3.722e-05, ..., 1.188e-04, 1.031e-03],
           [7.640e-05, 3.944e-05, ..., 5.180e-04, 1.346e-03]],
          dtype=float32)
    >>> phase
    array([[ 1.   +0.000e+00j,  1.   +0.000e+00j, ...,
            -1.   -8.742e-08j, -1.   -8.742e-08j],
           [-1.   -8.742e-08j, -0.775-6.317e-01j, ...,
            -0.885-4.648e-01j,  0.472-8.815e-01j],
           ...,
           [ 1.   -4.342e-12j,  0.028-9.996e-01j, ...,
            -0.222-9.751e-01j, -0.75 -6.610e-01j],
           [-1.   -8.742e-08j, -1.   -8.742e-08j, ...,
             1.   +0.000e+00j,  1.   +0.000e+00j]], dtype=complex64)

    Or get the phase angle (in radians)

    >>> np.angle(phase)
    array([[ 0.000e+00,  0.000e+00, ..., -3.142e+00, -3.142e+00],
           [-3.142e+00, -2.458e+00, ..., -2.658e+00, -1.079e+00],
           ...,
           [-4.342e-12, -1.543e+00, ..., -1.794e+00, -2.419e+00],
           [-3.142e+00, -3.142e+00, ...,  0.000e+00,  0.000e+00]],
          dtype=float32)
    """
    mag = np.abs(D)
    zeros_to_ones = mag == 0
    mag_nonzero = mag + zeros_to_ones
    phase = np.empty_like(D, dtype=util.dtype_r2c(D.dtype))
    phase.real = D.real / mag_nonzero + zeros_to_ones
    phase.imag = D.imag / mag_nonzero
    mag **= power
    return (mag, phase)