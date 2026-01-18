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
def spectral_bandwidth(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: int=512, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, pad_mode: _PadModeSTFT='constant', freq: Optional[np.ndarray]=None, centroid: Optional[np.ndarray]=None, norm: bool=True, p: float=2) -> np.ndarray:
    """Compute p'th-order spectral bandwidth.

       The spectral bandwidth [#]_ at frame ``t`` is computed by::

        (sum_k S[k, t] * (freq[k, t] - centroid[t])**p)**(1/p)

    .. [#] Klapuri, A., & Davy, M. (Eds.). (2007). Signal processing
        methods for music transcription, chapter 5.
        Springer Science & Business Media.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        audio sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) spectrogram magnitude
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
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
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    freq : None or np.ndarray [shape=(d,) or shape=(..., d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of ``d`` center frequencies,
        or a matrix of center frequencies as constructed by
        `librosa.reassigned_spectrogram`
    centroid : None or np.ndarray [shape=(..., 1, t)]
        pre-computed centroid frequencies
    norm : bool
        Normalize per-frame spectral energy (sum to one)
    p : float > 0
        Power to raise deviation from spectral centroid.

    Returns
    -------
    bandwidth : np.ndarray [shape=(..., 1, t)]
        frequency bandwidth for each frame

    Examples
    --------
    From time-series input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    >>> spec_bw
    array([[1273.836, 1228.873, ..., 2952.357, 3013.68 ]])

    From spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y=y))
    >>> librosa.feature.spectral_bandwidth(S=S)
    array([[1273.836, 1228.873, ..., 2952.357, 3013.68 ]])

    Using variable bin center frequencies

    >>> freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)
    >>> librosa.feature.spectral_bandwidth(S=np.abs(D), freq=freqs)
    array([[1274.637, 1228.786, ..., 2952.4  , 3013.735]])

    Plot the result

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> times = librosa.times_like(spec_bw)
    >>> centroid = librosa.feature.spectral_centroid(S=S)
    >>> ax[0].semilogy(times, spec_bw[0], label='Spectral bandwidth')
    >>> ax[0].set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
    >>> ax[0].legend()
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='log Power spectrogram')
    >>> ax[1].fill_between(times, np.maximum(0, centroid[0] - spec_bw[0]),
    ...                 np.minimum(centroid[0] + spec_bw[0], sr/2),
    ...                 alpha=0.5, label='Centroid +- bandwidth')
    >>> ax[1].plot(times, centroid[0], label='Spectral centroid', color='w')
    >>> ax[1].legend(loc='lower right')
    """
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    if not np.isrealobj(S):
        raise ParameterError('Spectral bandwidth is only defined with real-valued input')
    elif np.any(S < 0):
        raise ParameterError('Spectral bandwidth is only defined with non-negative energies')
    if centroid is None:
        centroid = spectral_centroid(y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, freq=freq)
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)
    if freq.ndim == 1:
        deviation = np.abs(np.subtract.outer(centroid[..., 0, :], freq).swapaxes(-2, -1))
    else:
        deviation = np.abs(freq - centroid)
    if norm:
        S = util.normalize(S, norm=1, axis=-2)
    bw: np.ndarray = np.sum(S * deviation ** p, axis=-2, keepdims=True) ** (1.0 / p)
    return bw