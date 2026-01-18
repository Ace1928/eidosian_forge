import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
from .. import util
from .. import filters
from ..util.exceptions import ParameterError
from ..core.convert import fft_frequencies
from ..core.audio import zero_crossings
from ..core.spectrum import power_to_db, _spectrogram
from ..core.constantq import cqt, hybrid_cqt, vqt
from ..core.pitch import estimate_tuning
from typing import Any, Optional, Union, Collection
from numpy.typing import DTypeLike
from .._typing import _FloatLike_co, _WindowSpec, _PadMode, _PadModeSTFT
def spectral_flatness(*, y: Optional[np.ndarray]=None, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: int=512, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, pad_mode: _PadModeSTFT='constant', amin: float=1e-10, power: float=2.0) -> np.ndarray:
    """Compute spectral flatness

    Spectral flatness (or tonality coefficient) is a measure to
    quantify how much noise-like a sound is, as opposed to being
    tone-like [#]_. A high spectral flatness (closer to 1.0)
    indicates the spectrum is similar to white noise.
    It is often converted to decibel.

    .. [#] Dubnov, Shlomo  "Generalization of spectral flatness
           measure for non-gaussian linear processes"
           IEEE Signal Processing Letters, 2004, Vol. 11.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) pre-computed spectrogram magnitude
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        .. see also:: `librosa.filters.get_window`
    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If `False`, then frame `t` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    amin : float > 0 [scalar]
        minimum threshold for ``S`` (=added noise floor for numerical stability)
    power : float > 0 [scalar]
        Exponent for the magnitude spectrogram.
        e.g., 1 for energy, 2 for power, etc.
        Power spectrogram is usually used for computing spectral flatness.

    Returns
    -------
    flatness : np.ndarray [shape=(..., 1, t)]
        spectral flatness for each frame.
        The returned value is in [0, 1] and often converted to dB scale.

    Examples
    --------
    From time-series input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> flatness = librosa.feature.spectral_flatness(y=y)
    >>> flatness
    array([[0.001, 0.   , ..., 0.218, 0.184]], dtype=float32)

    From spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> librosa.feature.spectral_flatness(S=S)
    array([[0.001, 0.   , ..., 0.218, 0.184]], dtype=float32)

    From power spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> S_power = S ** 2
    >>> librosa.feature.spectral_flatness(S=S_power, power=1.0)
    array([[0.001, 0.   , ..., 0.218, 0.184]], dtype=float32)

    """
    if amin <= 0:
        raise ParameterError('amin must be strictly positive')
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length, power=1.0, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    if not np.isrealobj(S):
        raise ParameterError('Spectral flatness is only defined with real-valued input')
    elif np.any(S < 0):
        raise ParameterError('Spectral flatness is only defined with non-negative energies')
    S_thresh = np.maximum(amin, S ** power)
    gmean = np.exp(np.mean(np.log(S_thresh), axis=-2, keepdims=True))
    amean = np.mean(S_thresh, axis=-2, keepdims=True)
    flatness: np.ndarray = gmean / amean
    return flatness