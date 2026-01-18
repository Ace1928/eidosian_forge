from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def rectified_complex_domain(spectrogram, diff_frames=None):
    """
    Rectified Complex Domain.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.
    diff_frames : int, optional
        Number of frames to calculate the diff to.

    Returns
    -------
    rectified_complex_domain : numpy array
        Rectified complex domain onset detection function.

    References
    ----------
    .. [1] Simon Dixon,
           "Onset Detection Revisited",
           Proceedings of the 9th International Conference on Digital Audio
           Effects (DAFx), 2006.

    """
    rcd = _complex_domain(spectrogram)
    pos_diff = spectrogram.diff(diff_frames=diff_frames, positive_diffs=True)
    rcd *= pos_diff.astype(bool)
    return np.asarray(np.sum(np.abs(rcd), axis=1))