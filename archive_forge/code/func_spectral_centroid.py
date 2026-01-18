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
def spectral_centroid(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: int=512, freq: Optional[np.ndarray]=None, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, pad_mode: _PadModeSTFT='constant') -> np.ndarray:
    """Compute the spectral centroid.

    Each frame of a magnitude spectrogram is normalized and treated as a
    distribution over frequency bins, from which the mean (centroid) is
    extracted per frame.

    More precisely, the centroid at frame ``t`` is defined as [#]_::

        centroid[t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t])

    where ``S`` is a magnitude spectrogram, and ``freq`` is the array of
    frequencies (e.g., FFT frequencies in Hz) of the rows of ``S``.

    .. [#] Klapuri, A., & Davy, M. (Eds.). (2007). Signal processing
        methods for music transcription, chapter 5.
        Springer Science & Business Media.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        audio sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) spectrogram magnitude
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of ``d`` center frequencies,
        or a matrix of center frequencies as constructed by
        `librosa.reassigned_spectrogram`
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length ``win_length`` and then padded
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
          `t` is centered at ``y[t * hop_length]``.
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    centroid : np.ndarray [shape=(..., 1, t)]
        centroid frequencies

    See Also
    --------
    librosa.stft : Short-time Fourier Transform
    librosa.reassigned_spectrogram : Time-frequency reassigned spectrogram

    Examples
    --------
    From time-series input:

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    >>> cent
    array([[1768.888, 1921.774, ..., 5663.477, 5813.683]])

    From spectrogram input:

    >>> S, phase = librosa.magphase(librosa.stft(y=y))
    >>> librosa.feature.spectral_centroid(S=S)
    array([[1768.888, 1921.774, ..., 5663.477, 5813.683]])

    Using variable bin center frequencies:

    >>> freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)
    >>> librosa.feature.spectral_centroid(S=np.abs(D), freq=freqs)
    array([[1768.838, 1921.801, ..., 5663.513, 5813.747]])

    Plot the result

    >>> import matplotlib.pyplot as plt
    >>> times = librosa.times_like(cent)
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax)
    >>> ax.plot(times, cent.T, label='Spectral centroid', color='w')
    >>> ax.legend(loc='upper right')
    >>> ax.set(title='log Power spectrogram')
    """
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    if not np.isrealobj(S):
        raise ParameterError('Spectral centroid is only defined with real-valued input')
    elif np.any(S < 0):
        raise ParameterError('Spectral centroid is only defined with non-negative energies')
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)
    if freq.ndim == 1:
        freq = util.expand_to(freq, ndim=S.ndim, axes=-2)
    centroid: np.ndarray = np.sum(freq * util.normalize(S, norm=1, axis=-2), axis=-2, keepdims=True)
    return centroid