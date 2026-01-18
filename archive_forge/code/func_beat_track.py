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
def beat_track(*, y: Optional[np.ndarray]=None, sr: float=22050, onset_envelope: Optional[np.ndarray]=None, hop_length: int=512, start_bpm: float=120.0, tightness: float=100, trim: bool=True, bpm: Optional[float]=None, prior: Optional[scipy.stats.rv_continuous]=None, units: str='frames') -> Tuple[float, np.ndarray]:
    """Dynamic programming beat tracker.

    Beats are detected in three stages, following the method of [#]_:

      1. Measure onset strength
      2. Estimate tempo from onset correlation
      3. Pick peaks in onset strength approximately consistent with estimated
         tempo

    .. [#] Ellis, Daniel PW. "Beat tracking by dynamic programming."
           Journal of New Music Research 36.1 (2007): 51-60.
           http://labrosa.ee.columbia.edu/projects/beattrack/

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series
    sr : number > 0 [scalar]
        sampling rate of ``y``
    onset_envelope : np.ndarray [shape=(n,)] or None
        (optional) pre-computed onset strength envelope.
    hop_length : int > 0 [scalar]
        number of audio samples between successive ``onset_envelope`` values
    start_bpm : float > 0 [scalar]
        initial guess for the tempo estimator (in beats per minute)
    tightness : float [scalar]
        tightness of beat distribution around tempo
    trim : bool [scalar]
        trim leading/trailing beats with weak onsets
    bpm : float [scalar]
        (optional) If provided, use ``bpm`` as the tempo instead of
        estimating it from ``onsets``.
    prior : scipy.stats.rv_continuous [optional]
        An optional prior distribution over tempo.
        If provided, ``start_bpm`` will be ignored.
    units : {'frames', 'samples', 'time'}
        The units to encode detected beat events in.
        By default, 'frames' are used.

    Returns
    -------
    tempo : float [scalar, non-negative]
        estimated global tempo (in beats per minute)
    beats : np.ndarray [shape=(m,)]
        estimated beat event locations in the specified units
        (default is frame indices)
    .. note::
        If no onset strength could be detected, beat_tracker estimates 0 BPM
        and returns an empty list.

    Raises
    ------
    ParameterError
        if neither ``y`` nor ``onset_envelope`` are provided,
        or if ``units`` is not one of 'frames', 'samples', or 'time'

    See Also
    --------
    librosa.onset.onset_strength

    Examples
    --------
    Track beats using time series input

    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)

    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> tempo
    135.99917763157896

    Print the frames corresponding to beats

    >>> beats
    array([  3,  21,  40,  59,  78,  96, 116, 135, 154, 173, 192, 211,
           230, 249, 268, 287, 306, 325, 344, 363])

    Or print them as timestamps

    >>> librosa.frames_to_time(beats, sr=sr)
    array([0.07 , 0.488, 0.929, 1.37 , 1.811, 2.229, 2.694, 3.135,
           3.576, 4.017, 4.458, 4.899, 5.341, 5.782, 6.223, 6.664,
           7.105, 7.546, 7.988, 8.429])

    Track beats using a pre-computed onset envelope

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          aggregate=np.median)
    >>> tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,
    ...                                        sr=sr)
    >>> tempo
    135.99917763157896
    >>> beats
    array([  3,  21,  40,  59,  78,  96, 116, 135, 154, 173, 192, 211,
           230, 249, 268, 287, 306, 325, 344, 363])

    Plot the beat events against the onset strength envelope

    >>> import matplotlib.pyplot as plt
    >>> hop_length = 512
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    >>> M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    >>> librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
    ...                          y_axis='mel', x_axis='time', hop_length=hop_length,
    ...                          ax=ax[0])
    >>> ax[0].label_outer()
    >>> ax[0].set(title='Mel spectrogram')
    >>> ax[1].plot(times, librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r',
    ...            linestyle='--', label='Beats')
    >>> ax[1].legend()
    """
    if onset_envelope is None:
        if y is None:
            raise ParameterError('y or onset_envelope must be provided')
        onset_envelope = onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)
    if not onset_envelope.any():
        return (0, np.array([], dtype=int))
    if bpm is None:
        bpm = _tempo(onset_envelope=onset_envelope, sr=sr, hop_length=hop_length, start_bpm=start_bpm, prior=prior)[0]
    beats = __beat_tracker(onset_envelope, bpm, float(sr) / hop_length, tightness, trim)
    if units == 'frames':
        return (bpm, beats)
    elif units == 'samples':
        return (bpm, core.frames_to_samples(beats, hop_length=hop_length))
    elif units == 'time':
        return (bpm, core.frames_to_time(beats, hop_length=hop_length, sr=sr))
    else:
        raise ParameterError(f'Invalid unit type: {units}')