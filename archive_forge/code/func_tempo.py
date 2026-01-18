import numpy as np
import scipy
from .. import util
from .._cache import cache
from ..core.audio import autocorrelate
from ..core.spectrum import stft
from ..core.convert import tempo_frequencies, time_to_frames
from ..core.harmonic import f0_harmonics
from ..util.exceptions import ParameterError
from ..filters import get_window
from typing import Optional, Callable, Any
from .._typing import _WindowSpec
@cache(level=30)
def tempo(*, y: Optional[np.ndarray]=None, sr: float=22050, onset_envelope: Optional[np.ndarray]=None, tg: Optional[np.ndarray]=None, hop_length: int=512, start_bpm: float=120, std_bpm: float=1.0, ac_size: float=8.0, max_tempo: Optional[float]=320.0, aggregate: Optional[Callable[..., Any]]=np.mean, prior: Optional[scipy.stats.rv_continuous]=None) -> np.ndarray:
    """Estimate the tempo (beats per minute)

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of the time series
    onset_envelope : np.ndarray [shape=(..., n)]
        pre-computed onset strength envelope
    tg : np.ndarray
        pre-computed tempogram.  If provided, then `y` and
        `onset_envelope` are ignored, and `win_length` is
        inferred from the shape of the tempogram.
    hop_length : int > 0 [scalar]
        hop length of the time series
    start_bpm : float [scalar]
        initial guess of the BPM
    std_bpm : float > 0 [scalar]
        standard deviation of tempo distribution
    ac_size : float > 0 [scalar]
        length (in seconds) of the auto-correlation window
    max_tempo : float > 0 [scalar, optional]
        If provided, only estimate tempo below this threshold
    aggregate : callable [optional]
        Aggregation function for estimating global tempo.
        If `None`, then tempo is estimated independently for each frame.
    prior : scipy.stats.rv_continuous [optional]
        A prior distribution over tempo (in beats per minute).
        By default, a pseudo-log-normal prior is used.
        If given, ``start_bpm`` and ``std_bpm`` will be ignored.

    Returns
    -------
    tempo : np.ndarray
        estimated tempo (beats per minute).
        If input is multi-channel, one tempo estimate per channel is provided.

    See Also
    --------
    librosa.onset.onset_strength
    librosa.feature.tempogram

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> # Estimate a static tempo
    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=30)
    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    >>> tempo
    array([143.555])

    >>> # Or a static tempo with a uniform prior instead
    >>> import scipy.stats
    >>> prior = scipy.stats.uniform(30, 300)  # uniform over 30-300 BPM
    >>> utempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, prior=prior)
    >>> utempo
    array([161.499])

    >>> # Or a dynamic tempo
    >>> dtempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr,
    ...                                aggregate=None)
    >>> dtempo
    array([ 89.103,  89.103,  89.103, ..., 123.047, 123.047, 123.047])

    >>> # Dynamic tempo with a proper log-normal prior
    >>> prior_lognorm = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
    >>> dtempo_lognorm = librosa.feature.tempo(onset_envelope=onset_env, sr=sr,
    ...                                        aggregate=None,
    ...                                        prior=prior_lognorm)
    >>> dtempo_lognorm
    array([ 89.103,  89.103,  89.103, ..., 123.047, 123.047, 123.047])

    Plot the estimated tempo against the onset autocorrelation

    >>> import matplotlib.pyplot as plt
    >>> # Convert to scalar
    >>> tempo = tempo.item()
    >>> utempo = utempo.item()
    >>> # Compute 2-second windowed autocorrelation
    >>> hop_length = 512
    >>> ac = librosa.autocorrelate(onset_env, max_size=2 * sr // hop_length)
    >>> freqs = librosa.tempo_frequencies(len(ac), sr=sr,
    ...                                   hop_length=hop_length)
    >>> # Plot on a BPM axis.  We skip the first (0-lag) bin.
    >>> fig, ax = plt.subplots()
    >>> ax.semilogx(freqs[1:], librosa.util.normalize(ac)[1:],
    ...              label='Onset autocorrelation', base=2)
    >>> ax.axvline(tempo, 0, 1, alpha=0.75, linestyle='--', color='r',
    ...             label='Tempo (default prior): {:.2f} BPM'.format(tempo))
    >>> ax.axvline(utempo, 0, 1, alpha=0.75, linestyle=':', color='g',
    ...             label='Tempo (uniform prior): {:.2f} BPM'.format(utempo))
    >>> ax.set(xlabel='Tempo (BPM)', title='Static tempo estimation')
    >>> ax.grid(True)
    >>> ax.legend()

    Plot dynamic tempo estimates over a tempogram

    >>> fig, ax = plt.subplots()
    >>> tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr,
    ...                                hop_length=hop_length)
    >>> librosa.display.specshow(tg, x_axis='time', y_axis='tempo', cmap='magma', ax=ax)
    >>> ax.plot(librosa.times_like(dtempo), dtempo,
    ...          color='c', linewidth=1.5, label='Tempo estimate (default prior)')
    >>> ax.plot(librosa.times_like(dtempo_lognorm), dtempo_lognorm,
    ...          color='c', linewidth=1.5, linestyle='--',
    ...          label='Tempo estimate (lognorm prior)')
    >>> ax.set(title='Dynamic tempo estimation')
    >>> ax.legend()
    """
    if start_bpm <= 0:
        raise ParameterError('start_bpm must be strictly positive')
    if tg is None:
        win_length = time_to_frames(ac_size, sr=sr, hop_length=hop_length).item()
        tg = tempogram(y=y, sr=sr, onset_envelope=onset_envelope, hop_length=hop_length, win_length=win_length)
    else:
        win_length = tg.shape[-2]
    if aggregate is not None:
        tg = aggregate(tg, axis=-1, keepdims=True)
    assert tg is not None
    bpms = tempo_frequencies(win_length, hop_length=hop_length, sr=sr)
    if prior is None:
        logprior = -0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm) ** 2
    else:
        logprior = prior.logpdf(bpms)
    if max_tempo is not None:
        max_idx = int(np.argmax(bpms < max_tempo))
        logprior[:max_idx] = -np.inf
    logprior = util.expand_to(logprior, ndim=tg.ndim, axes=-2)
    best_period = np.argmax(np.log1p(1000000.0 * tg) + logprior, axis=-2)
    tempo_est: np.ndarray = np.take(bpms, best_period)
    return tempo_est