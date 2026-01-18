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
def mu_compress(x: Union[np.ndarray, _FloatLike_co], *, mu: float=255, quantize: bool=True) -> np.ndarray:
    """mu-law compression

    Given an input signal ``-1 <= x <= 1``, the mu-law compression
    is calculated by::

        sign(x) * ln(1 + mu * abs(x)) /  ln(1 + mu)

    Parameters
    ----------
    x : np.ndarray with values in [-1, +1]
        The input signal to compress

    mu : positive number
        The compression parameter.  Values of the form ``2**n - 1``
        (e.g., 15, 31, 63, etc.) are most common.

    quantize : bool
        If ``True``, quantize the compressed values into ``1 + mu``
        distinct integer values.

        If ``False``, mu-law compression is applied without quantization.

    Returns
    -------
    x_compressed : np.ndarray
        The compressed signal.

    Raises
    ------
    ParameterError
        If ``x`` has values outside the range [-1, +1]
        If ``mu <= 0``

    See Also
    --------
    mu_expand

    Examples
    --------
    Compression without quantization

    >>> x = np.linspace(-1, 1, num=16)
    >>> x
    array([-1.        , -0.86666667, -0.73333333, -0.6       , -0.46666667,
           -0.33333333, -0.2       , -0.06666667,  0.06666667,  0.2       ,
            0.33333333,  0.46666667,  0.6       ,  0.73333333,  0.86666667,
            1.        ])
    >>> y = librosa.mu_compress(x, quantize=False)
    >>> y
    array([-1.        , -0.97430198, -0.94432361, -0.90834832, -0.86336132,
           -0.80328309, -0.71255496, -0.52124063,  0.52124063,  0.71255496,
            0.80328309,  0.86336132,  0.90834832,  0.94432361,  0.97430198,
            1.        ])

    Compression with quantization

    >>> y = librosa.mu_compress(x, quantize=True)
    >>> y
    array([-128, -124, -120, -116, -110, -102,  -91,  -66,   66,   91,  102,
           110,  116,  120,  124,  127])

    Compression with quantization and a smaller range

    >>> y = librosa.mu_compress(x, mu=15, quantize=True)
    >>> y
    array([-8, -7, -7, -6, -6, -5, -4, -2,  2,  4,  5,  6,  6,  7,  7,  7])
    """
    if mu <= 0:
        raise ParameterError(f'mu-law compression parameter mu={mu} must be strictly positive.')
    if np.any(x < -1) or np.any(x > 1):
        raise ParameterError(f'mu-law input x={x} must be in the range [-1, +1].')
    x_comp: np.ndarray = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    if quantize:
        y: np.ndarray = np.digitize(x_comp, np.linspace(-1, 1, num=int(1 + mu), endpoint=True), right=True) - int(mu + 1) // 2
        return y
    return x_comp