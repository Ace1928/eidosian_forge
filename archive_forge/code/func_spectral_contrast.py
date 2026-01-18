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
def spectral_contrast(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: int=512, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, pad_mode: _PadModeSTFT='constant', freq: Optional[np.ndarray]=None, fmin: float=200.0, n_bands: int=6, quantile: float=0.02, linear: bool=False) -> np.ndarray:
    """Compute spectral contrast

    Each frame of a spectrogram ``S`` is divided into sub-bands.
    For each sub-band, the energy contrast is estimated by comparing
    the mean energy in the top quantile (peak energy) to that of the
    bottom quantile (valley energy).  High contrast values generally
    correspond to clear, narrow-band signals, while low contrast values
    correspond to broad-band noise. [#]_

    .. [#] Jiang, Dan-Ning, Lie Lu, Hong-Jiang Zhang, Jian-Hua Tao,
           and Lian-Hong Cai.
           "Music type classification by spectral contrast feature."
           In Multimedia and Expo, 2002. ICME'02. Proceedings.
           2002 IEEE International Conference on, vol. 1, pp. 113-116.
           IEEE, 2002.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    sr : number  > 0 [scalar]
        audio sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) spectrogram magnitude
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
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    freq : None or np.ndarray [shape=(d,)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of ``d`` center frequencies.
    fmin : float > 0
        Frequency cutoff for the first bin ``[0, fmin]``
        Subsequent bins will cover ``[fmin, 2*fmin]`, `[2*fmin, 4*fmin]``, etc.
    n_bands : int > 1
        number of frequency bands
    quantile : float in (0, 1)
        quantile for determining peaks and valleys
    linear : bool
        If `True`, return the linear difference of magnitudes:
        ``peaks - valleys``.
        If `False`, return the logarithmic difference:
        ``log(peaks) - log(valleys)``.

    Returns
    -------
    contrast : np.ndarray [shape=(..., n_bands + 1, t)]
        each row of spectral contrast values corresponds to a given
        octave-based frequency

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img1 = librosa.display.specshow(librosa.amplitude_to_db(S,
    ...                                                  ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> fig.colorbar(img1, ax=[ax[0]], format='%+2.0f dB')
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> img2 = librosa.display.specshow(contrast, x_axis='time', ax=ax[1])
    >>> fig.colorbar(img2, ax=[ax[1]])
    >>> ax[1].set(ylabel='Frequency bands', title='Spectral contrast')
    """
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)
    freq = np.atleast_1d(freq)
    if freq.ndim != 1 or len(freq) != S.shape[-2]:
        raise ParameterError(f'freq.shape mismatch: expected ({S.shape[-2]:d},)')
    if n_bands < 1 or not isinstance(n_bands, (int, np.integer)):
        raise ParameterError('n_bands must be a positive integer')
    if not 0.0 < quantile < 1.0:
        raise ParameterError('quantile must lie in the range (0, 1)')
    if fmin <= 0:
        raise ParameterError('fmin must be a positive number')
    octa = np.zeros(n_bands + 2)
    octa[1:] = fmin * 2.0 ** np.arange(0, n_bands + 1)
    if np.any(octa[:-1] >= 0.5 * sr):
        raise ParameterError('Frequency band exceeds Nyquist. Reduce either fmin or n_bands.')
    shape = list(S.shape)
    shape[-2] = n_bands + 1
    valley = np.zeros(shape)
    peak = np.zeros_like(valley)
    for k, (f_low, f_high) in enumerate(zip(octa[:-1], octa[1:])):
        current_band = np.logical_and(freq >= f_low, freq <= f_high)
        idx = np.flatnonzero(current_band)
        if k > 0:
            current_band[idx[0] - 1] = True
        if k == n_bands:
            current_band[idx[-1] + 1:] = True
        sub_band = S[..., current_band, :]
        if k < n_bands:
            sub_band = sub_band[..., :-1, :]
        idx = np.rint(quantile * np.sum(current_band))
        idx = int(np.maximum(idx, 1))
        sortedr = np.sort(sub_band, axis=-2)
        valley[..., k, :] = np.mean(sortedr[..., :idx, :], axis=-2)
        peak[..., k, :] = np.mean(sortedr[..., -idx:, :], axis=-2)
    contrast: np.ndarray
    if linear:
        contrast = peak - valley
    else:
        contrast = power_to_db(peak) - power_to_db(valley)
    return contrast