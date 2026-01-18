from __future__ import annotations
import scipy.ndimage
import scipy.sparse
import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
from .._cache import cache
from .exceptions import ParameterError
from .deprecation import Deprecated
from numpy.typing import ArrayLike, DTypeLike
from typing import (
from typing_extensions import Literal
from .._typing import _SequenceLike, _FloatLike_co, _ComplexLike_co
def peak_pick(x: np.ndarray, *, pre_max: int, post_max: int, pre_avg: int, post_avg: int, delta: float, wait: int) -> np.ndarray:
    """Use a flexible heuristic to pick peaks in a signal.

    A sample n is selected as an peak if the corresponding ``x[n]``
    fulfills the following three conditions:

    1. ``x[n] == max(x[n - pre_max:n + post_max])``
    2. ``x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta``
    3. ``n - previous_n > wait``

    where ``previous_n`` is the last sample picked as a peak (greedily).

    This implementation is based on [#]_ and [#]_.

    .. [#] Boeck, Sebastian, Florian Krebs, and Markus Schedl.
        "Evaluating the Online Capabilities of Onset Detection Methods." ISMIR.
        2012.

    .. [#] https://github.com/CPJKU/onset_detection/blob/master/onset_program.py

    Parameters
    ----------
    x : np.ndarray [shape=(n,)]
        input signal to peak picks from
    pre_max : int >= 0 [scalar]
        number of samples before ``n`` over which max is computed
    post_max : int >= 1 [scalar]
        number of samples after ``n`` over which max is computed
    pre_avg : int >= 0 [scalar]
        number of samples before ``n`` over which mean is computed
    post_avg : int >= 1 [scalar]
        number of samples after ``n`` over which mean is computed
    delta : float >= 0 [scalar]
        threshold offset for mean
    wait : int >= 0 [scalar]
        number of samples to wait after picking a peak

    Returns
    -------
    peaks : np.ndarray [shape=(n_peaks,), dtype=int]
        indices of peaks in ``x``

    Raises
    ------
    ParameterError
        If any input lies outside its defined range

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          hop_length=512,
    ...                                          aggregate=np.median)
    >>> peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    >>> peaks
    array([  3,  27,  40,  61,  72,  88, 103])

    >>> import matplotlib.pyplot as plt
    >>> times = librosa.times_like(onset_env, sr=sr, hop_length=512)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> D = np.abs(librosa.stft(y))
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[0].plot(times, onset_env, alpha=0.8, label='Onset strength')
    >>> ax[0].vlines(times[peaks], 0,
    ...              onset_env.max(), color='r', alpha=0.8,
    ...              label='Selected peaks')
    >>> ax[0].legend(frameon=True, framealpha=0.8)
    >>> ax[0].label_outer()
    """
    if pre_max < 0:
        raise ParameterError('pre_max must be non-negative')
    if pre_avg < 0:
        raise ParameterError('pre_avg must be non-negative')
    if delta < 0:
        raise ParameterError('delta must be non-negative')
    if wait < 0:
        raise ParameterError('wait must be non-negative')
    if post_max <= 0:
        raise ParameterError('post_max must be positive')
    if post_avg <= 0:
        raise ParameterError('post_avg must be positive')
    if x.ndim != 1:
        raise ParameterError('input array must be one-dimensional')
    pre_max = valid_int(pre_max, cast=np.ceil)
    post_max = valid_int(post_max, cast=np.ceil)
    pre_avg = valid_int(pre_avg, cast=np.ceil)
    post_avg = valid_int(post_avg, cast=np.ceil)
    wait = valid_int(wait, cast=np.ceil)
    max_length = pre_max + post_max
    max_origin = np.ceil(0.5 * (pre_max - post_max))
    mov_max = scipy.ndimage.filters.maximum_filter1d(x, int(max_length), mode='constant', origin=int(max_origin), cval=x.min())
    avg_length = pre_avg + post_avg
    avg_origin = np.ceil(0.5 * (pre_avg - post_avg))
    mov_avg = scipy.ndimage.filters.uniform_filter1d(x, int(avg_length), mode='nearest', origin=int(avg_origin))
    n = 0
    while n - pre_avg < 0 and n < x.shape[0]:
        start = n - pre_avg
        start = start if start > 0 else 0
        mov_avg[n] = np.mean(x[start:n + post_avg])
        n += 1
    n = x.shape[0] - post_avg
    n = n if n > 0 else 0
    while n < x.shape[0]:
        start = n - pre_avg
        start = start if start > 0 else 0
        mov_avg[n] = np.mean(x[start:n + post_avg])
        n += 1
    detections = x * (x == mov_max)
    detections = detections * (detections >= mov_avg + delta)
    peaks = []
    last_onset = -np.inf
    for i in np.nonzero(detections)[0]:
        if i > last_onset + wait:
            peaks.append(i)
            last_onset = i
    return np.array(peaks)