from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def normalized_weighted_phase_deviation(spectrogram, epsilon=EPSILON):
    """
    Normalized Weighted Phase Deviation.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.
    epsilon : float, optional
        Add `epsilon` to the `spectrogram` avoid division by 0.

    Returns
    -------
    normalized_weighted_phase_deviation : numpy array
        Normalized weighted phase deviation onset detection function.

    References
    ----------
    .. [1] Simon Dixon,
           "Onset Detection Revisited",
           Proceedings of the 9th International Conference on Digital Audio
           Effects (DAFx), 2006.

    """
    if epsilon <= 0:
        raise ValueError('a positive value must be added before division')
    norm = np.add(np.mean(spectrogram, axis=1), epsilon)
    return np.asarray(weighted_phase_deviation(spectrogram) / norm)