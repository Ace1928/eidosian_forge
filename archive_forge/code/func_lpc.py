from __future__ import annotations
import os
import pathlib
import warnings
import soundfile as sf
import audioread
import numpy as np
import scipy.signal
import soxr
import lazy_loader as lazy
from numba import jit, stencil, guvectorize
from .fft import get_fftlib
from .convert import frames_to_samples, time_to_samples
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..util.decorators import deprecated
from ..util.deprecation import Deprecated, rename_kw
from .._typing import _FloatLike_co, _IntLike_co, _SequenceLike
from typing import Any, BinaryIO, Callable, Generator, Optional, Tuple, Union, List
from numpy.typing import DTypeLike, ArrayLike
def lpc(y: np.ndarray, *, order: int, axis: int=-1) -> np.ndarray:
    """Linear Prediction Coefficients via Burg's method

    This function applies Burg's method to estimate coefficients of a linear
    filter on ``y`` of order ``order``.  Burg's method is an extension to the
    Yule-Walker approach, which are both sometimes referred to as LPC parameter
    estimation by autocorrelation.

    It follows the description and implementation approach described in the
    introduction by Marple. [#]_  N.B. This paper describes a different method, which
    is not implemented here, but has been chosen for its clear explanation of
    Burg's technique in its introduction.

    .. [#] Larry Marple.
           A New Autoregressive Spectrum Analysis Algorithm.
           IEEE Transactions on Acoustics, Speech, and Signal Processing
           vol 28, no. 4, 1980.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        Time series to fit. Multi-channel is supported..
    order : int > 0
        Order of the linear filter
    axis : int
        Axis along which to compute the coefficients

    Returns
    -------
    a : np.ndarray [shape=(..., order + 1)]
        LP prediction error coefficients, i.e. filter denominator polynomial.
        Note that the length along the specified ``axis`` will be ``order+1``.

    Raises
    ------
    ParameterError
        - If ``y`` is not valid audio as per `librosa.util.valid_audio`
        - If ``order < 1`` or not integer
    FloatingPointError
        - If ``y`` is ill-conditioned

    See Also
    --------
    scipy.signal.lfilter

    Examples
    --------
    Compute LP coefficients of y at order 16 on entire series

    >>> y, sr = librosa.load(librosa.ex('libri1'))
    >>> librosa.lpc(y, order=16)

    Compute LP coefficients, and plot LP estimate of original series

    >>> import matplotlib.pyplot as plt
    >>> import scipy
    >>> y, sr = librosa.load(librosa.ex('libri1'), duration=0.020)
    >>> a = librosa.lpc(y, order=2)
    >>> b = np.hstack([[0], -1 * a[1:]])
    >>> y_hat = scipy.signal.lfilter(b, [1], y)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(y)
    >>> ax.plot(y_hat, linestyle='--')
    >>> ax.legend(['y', 'y_hat'])
    >>> ax.set_title('LP Model Forward Prediction')

    """
    if not util.is_positive_int(order):
        raise ParameterError(f'order={order} must be an integer > 0')
    util.valid_audio(y, mono=False)
    y = y.swapaxes(axis, 0)
    dtype = y.dtype
    shape = list(y.shape)
    shape[0] = order + 1
    ar_coeffs = np.zeros(tuple(shape), dtype=dtype)
    ar_coeffs[0] = 1
    ar_coeffs_prev = ar_coeffs.copy()
    shape[0] = 1
    reflect_coeff = np.zeros(shape, dtype=dtype)
    den = reflect_coeff.copy()
    epsilon = util.tiny(den)
    return np.swapaxes(__lpc(y, order, ar_coeffs, ar_coeffs_prev, reflect_coeff, den, epsilon), 0, axis)