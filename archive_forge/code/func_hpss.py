import numpy as np
import scipy.signal
from . import core
from . import decompose
from . import feature
from . import util
from .util.exceptions import ParameterError
from typing import Any, Callable, Iterable, Optional, Tuple, Union, overload
from typing_extensions import Literal
from numpy.typing import ArrayLike
def hpss(y: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose an audio time series into harmonic and percussive components.

    This function automates the STFT->HPSS->ISTFT pipeline, and ensures that
    the output waveforms have equal length to the input waveform ``y``.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    **kwargs : additional keyword arguments.
        See `librosa.decompose.hpss` for details.

    Returns
    -------
    y_harmonic : np.ndarray [shape=(..., n)]
        audio time series of the harmonic elements
    y_percussive : np.ndarray [shape=(..., n)]
        audio time series of the percussive elements

    See Also
    --------
    harmonic : Extract only the harmonic component
    percussive : Extract only the percussive component
    librosa.decompose.hpss : HPSS on spectrograms

    Examples
    --------
    >>> # Extract harmonic and percussive components
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> y_harmonic, y_percussive = librosa.effects.hpss(y)

    >>> # Get a more isolated percussive component by widening its margin
    >>> y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0,5.0))
    """
    stft = core.stft(y)
    stft_harm, stft_perc = decompose.hpss(stft, **kwargs)
    y_harm = core.istft(stft_harm, dtype=y.dtype, length=y.shape[-1])
    y_perc = core.istft(stft_perc, dtype=y.dtype, length=y.shape[-1])
    return (y_harm, y_perc)