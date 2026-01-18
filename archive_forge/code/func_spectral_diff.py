from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def spectral_diff(spectrogram, diff_frames=None):
    """
    Spectral Diff.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram instance.
    diff_frames : int, optional
        Number of frames to calculate the diff to.

    Returns
    -------
    spectral_diff : numpy array
        Spectral diff onset detection function.

    References
    ----------
    .. [1] Chris Duxbury, Mark Sandler and Matthew Davis,
           "A hybrid approach to musical note onset detection",
           Proceedings of the 5th International Conference on Digital Audio
           Effects (DAFx), 2002.

    """
    from madmom.audio.spectrogram import SpectrogramDifference
    if not isinstance(spectrogram, SpectrogramDifference):
        spectrogram = spectrogram.diff(diff_frames=diff_frames, positive_diffs=True)
    return np.asarray(np.sum(spectrogram ** 2, axis=1))