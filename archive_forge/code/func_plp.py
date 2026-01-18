import numpy as np
import scipy
import scipy.stats
from ._cache import cache
from . import core
from . import onset
from . import util
from .feature import tempogram, fourier_tempogram
from .feature import tempo as _tempo
from .util.exceptions import ParameterError
from .util.decorators import moved
from typing import Any, Callable, Optional, Tuple
def plp(*, y: Optional[np.ndarray]=None, sr: float=22050, onset_envelope: Optional[np.ndarray]=None, hop_length: int=512, win_length: int=384, tempo_min: Optional[float]=30, tempo_max: Optional[float]=300, prior: Optional[scipy.stats.rv_continuous]=None) -> np.ndarray:
    """Predominant local pulse (PLP) estimation. [#]_

    The PLP method analyzes the onset strength envelope in the frequency domain
    to find a locally stable tempo for each frame.  These local periodicities
    are used to synthesize local half-waves, which are combined such that peaks
    coincide with rhythmically salient frames (e.g. onset events on a musical time grid).
    The local maxima of the pulse curve can be taken as estimated beat positions.

    This method may be preferred over the dynamic programming method of `beat_track`
    when the tempo is expected to vary significantly over time.  Additionally,
    since `plp` does not require the entire signal to make predictions, it may be
    preferable when beat-tracking long recordings in a streaming setting.

    .. [#] Grosche, P., & Muller, M. (2011).
        "Extracting predominant local pulse information from music recordings."
        IEEE Transactions on Audio, Speech, and Language Processing, 19(6), 1688-1701.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    onset_envelope : np.ndarray [shape=(..., n)] or None
        (optional) pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        number of audio samples between successive ``onset_envelope`` values

    win_length : int > 0 [scalar]
        number of frames to use for tempogram analysis.
        By default, 384 frames (at ``sr=22050`` and ``hop_length=512``) corresponds
        to about 8.9 seconds.

    tempo_min, tempo_max : numbers > 0 [scalar], optional
        Minimum and maximum permissible tempo values.  ``tempo_max`` must be at least
        ``tempo_min``.

        Set either (or both) to `None` to disable this constraint.

    prior : scipy.stats.rv_continuous [optional]
        A prior distribution over tempo (in beats per minute).
        By default, a uniform prior over ``[tempo_min, tempo_max]`` is used.

    Returns
    -------
    pulse : np.ndarray, shape=[(..., n)]
        The estimated pulse curve.  Maxima correspond to rhythmically salient
        points of time.

        If input is multi-channel, one pulse curve per channel is computed.

    See Also
    --------
    beat_track
    librosa.onset.onset_strength
    librosa.feature.fourier_tempogram

    Examples
    --------
    Visualize the PLP compared to an onset strength envelope.
    Both are normalized here to make comparison easier.

    >>> y, sr = librosa.load(librosa.ex('brahms'))
    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    >>> # Or compute pulse with an alternate prior, like log-normal
    >>> import scipy.stats
    >>> prior = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
    >>> pulse_lognorm = librosa.beat.plp(onset_envelope=onset_env, sr=sr,
    ...                                  prior=prior)
    >>> melspec = librosa.feature.melspectrogram(y=y, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> librosa.display.specshow(librosa.power_to_db(melspec,
    ...                                              ref=np.max),
    ...                          x_axis='time', y_axis='mel', ax=ax[0])
    >>> ax[0].set(title='Mel spectrogram')
    >>> ax[0].label_outer()
    >>> ax[1].plot(librosa.times_like(onset_env),
    ...          librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[1].plot(librosa.times_like(pulse),
    ...          librosa.util.normalize(pulse),
    ...          label='Predominant local pulse (PLP)')
    >>> ax[1].set(title='Uniform tempo prior [30, 300]')
    >>> ax[1].label_outer()
    >>> ax[2].plot(librosa.times_like(onset_env),
    ...          librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[2].plot(librosa.times_like(pulse_lognorm),
    ...          librosa.util.normalize(pulse_lognorm),
    ...          label='Predominant local pulse (PLP)')
    >>> ax[2].set(title='Log-normal tempo prior, mean=120', xlim=[5, 20])
    >>> ax[2].legend()

    PLP local maxima can be used as estimates of beat positions.

    >>> tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
    >>> beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> times = librosa.times_like(onset_env, sr=sr)
    >>> ax[0].plot(times, librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[0].vlines(times[beats], 0, 1, alpha=0.5, color='r',
    ...            linestyle='--', label='Beats')
    >>> ax[0].legend()
    >>> ax[0].set(title='librosa.beat.beat_track')
    >>> ax[0].label_outer()
    >>> # Limit the plot to a 15-second window
    >>> times = librosa.times_like(pulse, sr=sr)
    >>> ax[1].plot(times, librosa.util.normalize(pulse),
    ...          label='PLP')
    >>> ax[1].vlines(times[beats_plp], 0, 1, alpha=0.5, color='r',
    ...            linestyle='--', label='PLP Beats')
    >>> ax[1].legend()
    >>> ax[1].set(title='librosa.beat.plp', xlim=[5, 20])
    >>> ax[1].xaxis.set_major_formatter(librosa.display.TimeFormatter())
    """
    if onset_envelope is None:
        onset_envelope = onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)
    if tempo_min is not None and tempo_max is not None and (tempo_max <= tempo_min):
        raise ParameterError(f'tempo_max={tempo_max} must be larger than tempo_min={tempo_min}')
    ftgram = fourier_tempogram(onset_envelope=onset_envelope, sr=sr, hop_length=hop_length, win_length=win_length)
    tempo_frequencies = core.fourier_tempo_frequencies(sr=sr, hop_length=hop_length, win_length=win_length)
    if tempo_min is not None:
        ftgram[..., tempo_frequencies < tempo_min, :] = 0
    if tempo_max is not None:
        ftgram[..., tempo_frequencies > tempo_max, :] = 0
    tempo_frequencies = util.expand_to(tempo_frequencies, ndim=ftgram.ndim, axes=-2)
    ftmag = np.log1p(1000000.0 * np.abs(ftgram))
    if prior is not None:
        ftmag += prior.logpdf(tempo_frequencies)
    peak_values = ftmag.max(axis=-2, keepdims=True)
    ftgram[ftmag < peak_values] = 0
    ftgram /= util.tiny(ftgram) ** 0.5 + np.abs(ftgram.max(axis=-2, keepdims=True))
    pulse = core.istft(ftgram, hop_length=1, n_fft=win_length, length=onset_envelope.shape[-1])
    pulse = np.clip(pulse, 0, None, pulse)
    return util.normalize(pulse, axis=-1)