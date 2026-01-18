from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def superflux(spectrogram, diff_frames=None, diff_max_bins=3):
    """
    SuperFlux method with a maximum filter vibrato suppression stage.

    Calculates the difference of bin k of the magnitude spectrogram relative to
    the N-th previous frame with the maximum filtered spectrogram.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram instance.
    diff_frames : int, optional
        Number of frames to calculate the diff to.
    diff_max_bins : int, optional
        Number of bins used for maximum filter.

    Returns
    -------
    superflux : numpy array
        SuperFlux onset detection function.

    Notes
    -----
    This method works only properly, if the spectrogram is filtered with a
    filterbank of the right frequency spacing. Filter banks with 24 bands per
    octave (i.e. quarter-tone resolution) usually yield good results. With
    `max_bins` = 3, the maximum of the bins k-1, k, k+1 of the frame
    `diff_frames` to the left is used for the calculation of the difference.

    References
    ----------
    .. [1] Sebastian Böck and Gerhard Widmer,
           "Maximum Filter Vibrato Suppression for Onset Detection",
           Proceedings of the 16th International Conference on Digital Audio
           Effects (DAFx), 2013.

    """
    from madmom.audio.spectrogram import SpectrogramDifference
    if not isinstance(spectrogram, SpectrogramDifference):
        spectrogram = spectrogram.diff(diff_frames=diff_frames, diff_max_bins=diff_max_bins, positive_diffs=True)
    return np.asarray(np.sum(spectrogram, axis=1))