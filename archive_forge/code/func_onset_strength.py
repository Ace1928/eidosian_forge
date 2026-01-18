import numpy as np
import scipy
from ._cache import cache
from . import core
from . import util
from .util.exceptions import ParameterError
from .feature.spectral import melspectrogram
from typing import Any, Callable, Iterable, Optional, Union, Sequence
def onset_strength(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, lag: int=1, max_size: int=1, ref: Optional[np.ndarray]=None, detrend: bool=False, center: bool=True, feature: Optional[Callable]=None, aggregate: Optional[Union[Callable, bool]]=None, **kwargs: Any) -> np.ndarray:
    """Compute a spectral flux onset strength envelope.

    Onset strength at time ``t`` is determined by::

        mean_f max(0, S[f, t] - ref[f, t - lag])

    where ``ref`` is ``S`` after local max filtering along the frequency
    axis [#]_.

    By default, if a time series ``y`` is provided, S will be the
    log-power Mel spectrogram.

    .. [#] Böck, Sebastian, and Gerhard Widmer.
           "Maximum filter vibrato suppression for onset detection."
           16th International Conference on Digital Audio Effects,
           Maynooth, Ireland. 2013.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time-series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, m)]
        pre-computed (log-power) spectrogram

    lag : int > 0
        time lag for computing differences

    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.

    ref : None or np.ndarray [shape=(..., d, m)]
        An optional pre-computed reference spectrum, of the same shape as ``S``.
        If not provided, it will be computed from ``S``.
        If provided, it will override any local max filtering governed by ``max_size``.

    detrend : bool [scalar]
        Filter the onset strength to remove the DC component

    center : bool [scalar]
        Shift the onset function by ``n_fft // (2 * hop_length)`` frames.
        This corresponds to using a centered frame analysis in the short-time Fourier
        transform.

    feature : function
        Function for computing time-series features, eg, scaled spectrograms.
        By default, uses `librosa.feature.melspectrogram` with ``fmax=sr/2``

    aggregate : function
        Aggregation function to use when combining onsets
        at different frequency bins.

        Default: `np.mean`

    **kwargs : additional keyword arguments
        Additional parameters to ``feature()``, if ``S`` is not provided.

    Returns
    -------
    onset_envelope : np.ndarray [shape=(..., m,)]
        vector containing the onset strength envelope.
        If the input contains multiple channels, then onset envelope is computed for each channel.

    Raises
    ------
    ParameterError
        if neither ``(y, sr)`` nor ``S`` are provided

        or if ``lag`` or ``max_size`` are not positive integers

    See Also
    --------
    onset_detect
    onset_strength_multi

    Examples
    --------
    First, load some audio and plot the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> D = np.abs(librosa.stft(y))
    >>> times = librosa.times_like(D)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()

    Construct a standard onset function

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> ax[1].plot(times, 2 + onset_env / onset_env.max(), alpha=0.8,
    ...            label='Mean (mel)')

    Median aggregation, and custom mel options

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          aggregate=np.median,
    ...                                          fmax=8000, n_mels=256)
    >>> ax[1].plot(times, 1 + onset_env / onset_env.max(), alpha=0.8,
    ...            label='Median (custom mel)')

    Constant-Q spectrogram instead of Mel

    >>> C = np.abs(librosa.cqt(y=y, sr=sr))
    >>> onset_env = librosa.onset.onset_strength(sr=sr, S=librosa.amplitude_to_db(C, ref=np.max))
    >>> ax[1].plot(times, onset_env / onset_env.max(), alpha=0.8,
    ...          label='Mean (CQT)')
    >>> ax[1].legend()
    >>> ax[1].set(ylabel='Normalized strength', yticks=[])
    """
    if aggregate is False:
        raise ParameterError(f'aggregate parameter cannot be False when computing full-spectrum onset strength.')
    odf_all = onset_strength_multi(y=y, sr=sr, S=S, lag=lag, max_size=max_size, ref=ref, detrend=detrend, center=center, feature=feature, aggregate=aggregate, channels=None, **kwargs)
    return odf_all[..., 0, :]