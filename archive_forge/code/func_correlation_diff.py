from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def correlation_diff(spec, diff_frames=1, pos=False, diff_bins=1):
    """
    Calculates the difference of the magnitude spectrogram relative to the
    N-th previous frame shifted in frequency to achieve the highest
    correlation between these two frames.

    Parameters
    ----------
    spec : numpy array
        Magnitude spectrogram.
    diff_frames : int, optional
        Calculate the difference to the `diff_frames`-th previous frame.
    pos : bool, optional
        Keep only positive values.
    diff_bins : int, optional
        Maximum number of bins shifted for correlation calculation.

    Returns
    -------
    correlation_diff : numpy array
        (Positive) magnitude spectrogram differences.

    Notes
    -----
    This function is only because of completeness, it is not intended to be
    actually used, since it is extremely slow. Please consider the superflux()
    function, since if performs equally well but much faster.

    """
    diff_spec = np.zeros_like(spec)
    if diff_frames < 1:
        raise ValueError('number of `diff_frames` must be >= 1')
    frames, bins = diff_spec.shape
    corr = np.zeros((frames, diff_bins * 2 + 1))
    for f in range(diff_frames, frames):
        c = np.correlate(spec[f], spec[f - diff_frames], mode='full')
        centre = len(c) / 2
        corr[f] = c[centre - diff_bins:centre + diff_bins + 1]
        bin_offset = diff_bins - np.argmax(corr[f])
        bin_start = diff_bins + bin_offset
        bin_stop = bins - 2 * diff_bins + bin_start
        diff_spec[f, diff_bins:-diff_bins] = spec[f, diff_bins:-diff_bins] - spec[f - diff_frames, bin_start:bin_stop]
    if pos:
        np.maximum(diff_spec, 0, diff_spec)
    return np.asarray(diff_spec)